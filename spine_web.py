"""预览用的本地小服务。

渲染换成网页里的 pixi-spine(WebGL)之后,得有个地方让浏览器拿到素材和把导出结果送回来:
这里起一个只监听 127.0.0.1 的小 http 服务,挂上打包进 exe 的网页,再用系统浏览器
的 app 模式开窗口。导出时网页把逐帧 PNG 传进来,这边用 ffmpeg 合成 mp4。
"""
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse

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

    def __init__(self, ask_path=None):
        self.ask_path = ask_path      # 由主线程弹保存对话框,签名 ask_path(anim) -> 路径或 ''
        self.res_root = None
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

    def set_res_root(self, directory):
        self.res_root = os.path.abspath(directory)

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


def open_viewer(server, json_path):
    """开预览窗口;素材目录换成这份骨架所在目录"""
    server.ensure_started()
    server.set_res_root(os.path.dirname(os.path.abspath(json_path)))
    server.default_name = os.path.splitext(os.path.basename(json_path))[0]
    url = 'http://127.0.0.1:%d/?json=%s' % (
        server.port, urllib.parse.quote(os.path.basename(json_path)))
    exe = next((p for p in _BROWSERS if os.path.isfile(p)), None)
    if exe:
        try:
            subprocess.Popen([exe, '--app=' + url, '--window-size=1180,860'])
            return url
        except Exception:
            pass
    import webbrowser
    webbrowser.open(url)
    return url