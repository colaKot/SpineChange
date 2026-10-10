"""预览用的本地小服务。

渲染换成网页里的 pixi-spine(WebGL)之后,得有个地方让浏览器拿到素材和把导出结果送回来:
这里起一个只监听 127.0.0.1 的小 http 服务,挂上打包进 exe 的网页,再用系统浏览器
的 app 模式开窗口。导出时网页把逐帧 PNG 传进来,这边用 ffmpeg 合成 mp4。
"""
import ctypes
import ctypes.wintypes
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid

import numpy as np
from PIL import Image

_BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def _web_root():
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'web')


def _ctype(full):
    low = full.lower()
    if low.endswith('.html'):
        return 'text/html; charset=utf-8'
    if low.endswith('.js'):
        return 'application/javascript; charset=utf-8'
    if low.endswith('.png'):
        return 'image/png'
    if low.endswith('.json'):
        return 'application/json; charset=utf-8'
    if low.endswith('.atlas'):
        return 'text/plain; charset=utf-8'
    return 'application/octet-stream'


class ViewerServer:
    """进程内单例:静态资源 + 导出回调"""

    def __init__(self, ask_path=None, loader=None):
        self.ask_path = ask_path      # 由主线程弹保存对话框,签名 ask_path(anim) -> 路径或 ''
        self.loader = loader          # 把任意版本的骨架读成 3.8 字典,签名 loader(path) -> dict
        self.res_root = None
        self.skel_path = None
        self.default_name = 'spine'
        self.port = None
        self._httpd = None
        self._frames_dir = None
        self._frames = 0

    # ---------- 生命周期 ----------
    def ensure_started(self):
        if self._httpd is not None:
            return
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype='application/json; charset=utf-8'):
                if isinstance(body, str):
                    body = body.encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def _file(self, full):
                if not os.path.isfile(full):
                    return self._send(404, 'not found')
                with open(full, 'rb') as f:
                    self._send(200, f.read(), _ctype(full))

            def do_GET(self):
                path = urllib.parse.urlparse(self.path).path
                if path in ('/', '/index.html'):
                    return self._file(os.path.join(_web_root(), 'spine_viewer.html'))
                if path.startswith('/lib/'):
                    return self._file(os.path.join(_web_root(), urllib.parse.unquote(path[1:])))
                if path.startswith('/api/skeleton/'):
                    name = urllib.parse.unquote(path[len('/api/skeleton/'):])
                    # 网页只认 3.8,所以骨架统一转好再给;.atlas/.png 这些贴图资源照原样给
                    if name.lower().endswith('.json'):
                        if server.loader is None or not server.skel_path:
                            return self._send(404, 'no skeleton')
                        try:
                            data = server.loader(server.skel_path)
                        except Exception as e:
                            return self._send(500, '骨架转换失败: %s' % e,
                                              'text/plain; charset=utf-8')
                        return self._send(200, json.dumps(data, ensure_ascii=False),
                                          'application/json; charset=utf-8')
                    if server.res_root is None:
                        return self._send(404, 'no resource')
                    full = os.path.normpath(os.path.join(server.res_root, name))
                    if not full.startswith(server.res_root):
                        return self._send(403, 'denied')
                    return self._file(full)
                if path.startswith('/res/'):
                    if server.res_root is None:
                        return self._send(404, 'no resource')
                    full = os.path.normpath(
                        os.path.join(server.res_root, urllib.parse.unquote(path[5:])))
                    if not full.startswith(server.res_root):
                        return self._send(403, 'denied')
                    return self._file(full)
                self._send(404, 'not found')

            def do_POST(self):
                url = urllib.parse.urlparse(self.path)
                if url.path == '/api/pick_save':
                    anim = urllib.parse.parse_qs(url.query).get('name', ['anim'])[0]
                    path = server.ask_path(anim) if server.ask_path else ''
                    return self._send(200, json.dumps({'path': path or ''}))
                if url.path == '/api/frame':
                    n = int(self.headers.get('Content-Length') or 0)
                    server.save_frame(self.rfile.read(n) if n else b'')
                    return self._send(200, 'ok', 'text/plain')
                if url.path == '/api/video_end':
                    n = int(self.headers.get('Content-Length') or 0)
                    info = json.loads(self.rfile.read(n).decode('utf-8')) if n else {}
                    err = server.finish_video(info)
                    if err:
                        return self._send(200, json.dumps({'error': err}))
                    return self._send(200, json.dumps({'ok': True, 'path': info.get('path')}))
                self._send(404, 'not found')

        self._httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def shutdown(self):
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            self._httpd = None

    def set_asset(self, path):
        """预览换一份素材:骨架路径、图集目录、默认文件名都跟着走"""
        path = os.path.abspath(path)
        self.skel_path = path
        self.res_root = os.path.dirname(path)
        self.default_name = os.path.splitext(os.path.basename(path))[0]

    # ---------- 收帧 / 合成 ----------
    def save_frame(self, data):
        if not data:
            return
        if self._frames_dir is None:
            self._frames_dir = tempfile.mkdtemp(prefix='spine_video_')
            self._frames = 0
        with open(os.path.join(self._frames_dir, 'f_%06d.png' % self._frames), 'wb') as f:
            f.write(data)
        self._frames += 1

    def finish_video(self, info):
        path = info.get('path')
        fps = float(info.get('fps') or 30)
        frames = int(info.get('frames') or 0)
        tmpdir = self._frames_dir
        writer = None
        try:
            if not path:
                return '没有保存路径'
            import imageio
            writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8,
                                        pixelformat='yuv420p', macro_block_size=None)
            for i in range(frames):
                fn = os.path.join(tmpdir, 'f_%06d.png' % i)
                if not os.path.isfile(fn):
                    continue
                writer.append_data(np.asarray(Image.open(fn).convert('RGB')))
            return None
        except Exception as e:
            return str(e)
        finally:
            if writer is not None:
                # ffmpeg 已经挂了的话 close 会再抛一次,别把真正的错因盖掉
                try:
                    writer.close()
                except Exception:
                    pass
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            self._frames_dir = None
            self._frames = 0


# ======================== 内嵌浏览器 ========================

_GWL_STYLE = -16
_WS_CHILD = 0x40000000
_WS_POPUP = 0x80000000
_WS_CAPTION = 0x00C00000
_WS_THICKFRAME = 0x00040000

_user32_cache = None


def _user32():
    """配好参数类型的 user32,不然后面传 HWND 在 64 位上会被截断"""
    global _user32_cache
    if _user32_cache is not None:
        return _user32_cache
    u = ctypes.windll.user32
    u.SetParent.restype = ctypes.c_void_p
    u.SetParent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    u.GetWindowLongW.restype = ctypes.c_long
    u.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    u.SetWindowLongW.restype = ctypes.c_long
    u.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    u.MoveWindow.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_bool]
    u.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.wintypes.RECT)]
    u.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.wintypes.RECT)]
    u.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    _user32_cache = u
    return u


class EmbeddedBrowser:
    """把浏览器窗口挂进 tk 的 frame 里,看起来就跟界面是一体的

    tk 自己画不了 WebGL,只能借系统浏览器;这里开个 --app 窗口,再把它的 HWND
    用 SetParent 挂到 frame 上。挂不上就留在外面当普通窗口,不影响用。
    """

    def __init__(self, url, parent_hwnd, size=(1180, 760)):
        self.url = url
        self.parent = int(parent_hwnd)
        self.size = size
        self.title = 'SpineChange预览-' + uuid.uuid4().hex[:8]
        self.proc = None
        self.hwnd = None

    def start(self):
        exe = next((p for p in _BROWSERS if os.path.isfile(p)), None)
        if not exe:
            return False
        left, top, w, h = self._parent_rect()
        cmd = [exe,
               '--app=%s&title=%s' % (self.url, urllib.parse.quote(self.title)),
               '--user-data-dir=' + tempfile.mkdtemp(prefix='spine_web_'),
               '--no-first-run', '--no-default-browser-check',
               '--window-position=%d,%d' % (left, top),
               '--window-size=%d,%d' % (w, h)]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        except Exception:
            return False
        # 等窗口起来再挂,别卡住 tk 主线程
        threading.Thread(target=self._attach, daemon=True).start()
        return True

    def _parent_rect(self):
        u = _user32()
        rect = ctypes.wintypes.RECT()
        if u.GetWindowRect(ctypes.c_void_p(self.parent), ctypes.byref(rect)):
            w = max(320, rect.right - rect.left)
            h = max(240, rect.bottom - rect.top)
            return rect.left, rect.top, w, h
        return 80, 80, self.size[0], self.size[1]

    def _attach(self):
        u = _user32()
        end = time.time() + 12
        while time.time() < end:
            hwnd = self._find_window(u)
            if hwnd:
                try:
                    # 必须先改成 WS_CHILD 再 SetParent:顺序反了 SetParent 不生效
                    style = u.GetWindowLongW(hwnd, _GWL_STYLE)
                    style &= ~(_WS_POPUP | _WS_CAPTION | _WS_THICKFRAME)
                    style |= _WS_CHILD
                    style &= 0xFFFFFFFF
                    if style >= 0x80000000:
                        style -= 0x100000000
                    u.SetWindowLongW(hwnd, _GWL_STYLE, style)
                    u.SetParent(hwnd, ctypes.c_void_p(self.parent))
                    self.hwnd = hwnd
                    self.resize()
                except Exception:
                    self.hwnd = None
                return
            time.sleep(0.25)

    def _find_window(self, u):
        found = []

        def cb(hwnd, _param):
            buf = ctypes.create_unicode_buffer(256)
            u.GetWindowTextW(hwnd, buf, 256)
            if buf.value == self.title:
                found.append(hwnd)
            return True

        u.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p,
                                         ctypes.c_void_p)(cb), None)
        return found[0] if found else None

    def resize(self):
        if not self.hwnd:
            return
        u = _user32()
        rect = ctypes.wintypes.RECT()
        if not u.GetClientRect(ctypes.c_void_p(self.parent), ctypes.byref(rect)):
            return
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w > 0 and h > 0:
            u.MoveWindow(self.hwnd, 0, 0, w, h, True)

    def close(self):
        self.hwnd = None
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None


def _open_browser_window(url):
    exe = next((p for p in _BROWSERS if os.path.isfile(p)), None)
    if exe:
        try:
            subprocess.Popen([exe, '--app=' + url, '--window-size=1180,860'])
            return
        except Exception:
            pass
    import webbrowser
    webbrowser.open(url)


def open_viewer(server, json_path, parent_hwnd=None, src_version='', request_name=None):
    """打开预览:能挂进 tk 窗口就挂进去,挂不了再退回独立窗口

    request_name 是网页要取的骨架名:老版本给 .json(服务端转 3.8),
    本身 3.8 的二进制给 .skel(直接读)。不传就按 .json 处理。
    返回内嵌的浏览器对象(供上层做缩放/关闭),退回独立窗口时返回 None。
    """
    server.ensure_started()
    server.set_asset(json_path)
    name = request_name or (server.default_name + '.json')
    url = 'http://127.0.0.1:%d/?json=%s&ver=%s' % (
        server.port, urllib.parse.quote(name), urllib.parse.quote(src_version or ''))
    if parent_hwnd:
        browser = EmbeddedBrowser(url, parent_hwnd)
        browser.start()
        return browser
    _open_browser_window(url)
    return None