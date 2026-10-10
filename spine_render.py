"""Spine 3.8 动画运行时 + 渲染器(纯 Python + Pillow + numpy)

不依赖任何官方 runtime,自己按 spine-runtimes 3.8 的逻辑实现:
骨骼世界变换、时间轴采样、IK/Transform/Path 约束、区域/网格附件绘制。
渲染走 numpy 软件光栅化,最终输出 PIL.Image。
"""
import os
import math
import json

import numpy as np
from PIL import Image

D2R = math.pi / 180.0

# 变换模式
TM_NORMAL, TM_ONLY_TRANSLATION, TM_NO_ROT_TRANS, TM_NO_SCALE_REFL = 0, 1, 2, 3
_TRANSFORM_MODE = {
    "normal": TM_NORMAL,
    "onlyTranslation": TM_ONLY_TRANSLATION,
    "noRotationOrReflection": TM_NO_ROT_TRANS,
    "noScaleOrReflection": TM_NO_SCALE_REFL,
}

# 混合模式
BLEND_NORMAL, BLEND_ADDITIVE, BLEND_MULTIPLY, BLEND_SCREEN = 0, 1, 2, 3
_BLEND_MODE = {"normal": BLEND_NORMAL, "additive": BLEND_ADDITIVE,
               "multiply": BLEND_MULTIPLY, "screen": BLEND_SCREEN}


# ============================ 曲线采样 ============================

def _bez(t, p1, p2):
    mt = 1.0 - t
    return 3.0 * mt * mt * t * p1 + 3.0 * mt * t * t * p2 + t * t * t


def _bezier(pct, cx1, cy1, cx2, cy2):
    if pct <= 0.0:
        return 0.0
    if pct >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(22):
        t = (lo + hi) * 0.5
        if _bez(t, cx1, cx2) < pct:
            lo = t
        else:
            hi = t
    return _bez((lo + hi) * 0.5, cy1, cy2)


def _curve_pct(frame, pct):
    """把线性进度按当前帧的曲线换算成插值进度"""
    c = frame.get('curve')
    if c is None:
        return pct
    if c == 'stepped' or c == 'stepped ':
        return 0.0
    if isinstance(c, (list, tuple)):
        if len(c) >= 4:
            return _bezier(pct, c[0], c[1], c[2], c[3])
        return pct
    cx1 = float(c)
    cy1 = float(frame.get('c2', cx1))
    cx2 = float(frame.get('c3', cy1))
    cy2 = float(frame.get('c4', cx2))
    return _bezier(pct, cx1, cy1, cx2, cy2)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _ftime(f):
    # spine 的 JSON 会省略 time=0 的首帧
    t = f.get('time')
    return 0.0 if t is None else t


def _find_frames(frames, time):
    """返回 (i, 是否处于两帧之间)"""
    n = len(frames)
    if n == 1 or time <= _ftime(frames[0]):
        return 0, False
    if time >= _ftime(frames[n - 1]):
        return n - 1, False
    lo, hi = 0, n - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if _ftime(frames[mid]) <= time:
            lo = mid
        else:
            hi = mid
    return lo, True


def sample(frames, time, keys, defaults):
    """采样一条时间轴,返回 dict"""
    if not frames:
        return dict(defaults)
    i, between = _find_frames(frames, time)
    f0 = frames[i]
    out = {}
    if not between:
        for k in keys:
            out[k] = f0.get(k, defaults.get(k))
        return out
    f1 = frames[i + 1]
    span = _ftime(f1) - _ftime(f0)
    pct = 0.0 if span <= 0 else (time - _ftime(f0)) / span
    pct = _curve_pct(f0, pct)
    for k in keys:
        a = f0.get(k, defaults.get(k))
        b = f1.get(k, defaults.get(k))
        if a is None:
            a = b
        if b is None:
            b = a
        if _is_num(a) and _is_num(b):
            out[k] = a + (b - a) * pct
        else:
            # 颜色、布尔等非数值只能整帧切换
            out[k] = a if pct < 1.0 else b
    return out


# ============================ 图集 ============================

class AtlasRegion:
    __slots__ = ('name', 'page', 'x', 'y', 'w', 'h', 'rotate',
                 'orig_w', 'orig_h', 'off_x', 'off_y', 'index')

    def __init__(self):
        self.name = ''
        self.page = None
        self.x = self.y = 0
        self.w = self.h = 0
        self.rotate = False
        self.orig_w = self.orig_h = 0
        self.off_x = self.off_y = 0
        self.index = -1


def _parse_int_pair(v, default=(0, 0)):
    try:
        parts = [p.strip() for p in v.split(',')]
        return int(parts[0]), int(parts[1])
    except Exception:
        return default


def load_atlas(atlas_path):
    """解析 .atlas,返回 (regions: {name: [AtlasRegion]}, pages: {name: 图片文件名})"""
    with open(atlas_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    lines = [ln.rstrip('\r\n') for ln in raw.split('\n')]

    # 图集格式里 page 的属性和 region 名都可能顶格,没法只看缩进;
    # 可靠的判断是:不含冒号的行是新条目(page 文件名或 region 名),含冒号的是属性
    entries = []
    head = None
    attrs = {}
    for ln in lines:
        if not ln.strip():
            continue
        if ':' not in ln:
            if head is not None:
                entries.append((head, attrs))
            head = ln.strip()
            attrs = {}
        else:
            k, v = ln.split(':', 1)
            attrs[k.strip()] = v.strip()
    if head is not None:
        entries.append((head, attrs))

    regions = {}
    pages = {}
    page_file = None
    for head, attrs in entries:
        is_page = ('format' in attrs) or ('filter' in attrs) or ('repeat' in attrs)
        if is_page:
            page_file = head
            pages[head] = head
            continue
        if page_file is None:
            continue
        r = AtlasRegion()
        r.name = head
        r.page = page_file
        if 'xy' in attrs:
            r.x, r.y = _parse_int_pair(attrs['xy'])
        if 'size' in attrs:
            r.w, r.h = _parse_int_pair(attrs['size'])
        if 'orig' in attrs:
            r.orig_w, r.orig_h = _parse_int_pair(attrs['orig'])
        else:
            r.orig_w, r.orig_h = r.w, r.h
        if 'offset' in attrs:
            r.off_x, r.off_y = _parse_int_pair(attrs['offset'])
        if 'rotate' in attrs:
            r.rotate = attrs['rotate'].strip().lower() in ('true', '90', 'yes')
        if 'index' in attrs:
            try:
                r.index = int(attrs['index'])
            except Exception:
                r.index = -1
        regions.setdefault(r.name, []).append(r)
    return regions, pages


# ============================ 骨骼 ============================

class Bone:
    __slots__ = ('name', 'parent', 'length', 'setup', 'transform_mode',
                 'rotation', 'x', 'y', 'scale_x', 'scale_y', 'shear_x', 'shear_y',
                 'a', 'b', 'c', 'd', 'world_x', 'world_y',
                 'world_rotation', 'world_scale_x', 'world_scale_y', 'children')

    def __init__(self, data, parent):
        self.name = data.get('name')
        self.parent = parent
        self.length = float(data.get('length', 0.0) or 0.0)
        self.setup = (
            float(data.get('rotation', 0.0) or 0.0),
            float(data.get('x', 0.0) or 0.0),
            float(data.get('y', 0.0) or 0.0),
            float(data.get('scaleX', 1.0) or 1.0),
            float(data.get('scaleY', 1.0) or 1.0),
            float(data.get('shearX', 0.0) or 0.0),
            float(data.get('shearY', 0.0) or 0.0),
        )
        self.transform_mode = _TRANSFORM_MODE.get(data.get('transform', 'normal'), TM_NORMAL)
        self.children = []
        if parent is not None:
            parent.children.append(self)
        self.reset()

    def reset(self):
        (self.rotation, self.x, self.y, self.scale_x, self.scale_y,
         self.shear_x, self.shear_y) = self.setup
        self.a, self.b, self.c, self.d = 1.0, 0.0, 0.0, 1.0
        self.world_x = self.world_y = 0.0
        self.world_rotation = 0.0
        self.world_scale_x = self.world_scale_y = 1.0

    def update(self):
        p = self.parent
        if p is None:
            pa, pb, pc, pd = 1.0, 0.0, 0.0, 1.0
            px = py = 0.0
        else:
            pa, pb, pc, pd = p.a, p.b, p.c, p.d
            px, py = p.world_x, p.world_y

        rot = self.rotation
        sx, sy = self.scale_x, self.scale_y
        rot_y = rot + 90.0 + self.shear_y
        la = math.cos((rot + self.shear_x) * D2R) * sx
        lb = math.cos(rot_y * D2R) * sy
        lc = math.sin((rot + self.shear_x) * D2R) * sx
        ld = math.sin(rot_y * D2R) * sy

        tm = self.transform_mode
        if tm == TM_NORMAL:
            self.world_x = self.x * pa + self.y * pb + px
            self.world_y = self.x * pc + self.y * pd + py
            self.a = pa * la + pb * lc
            self.b = pa * lb + pb * ld
            self.c = pc * la + pd * lc
            self.d = pc * lb + pd * ld
        elif tm == TM_ONLY_TRANSLATION:
            self.a, self.b, self.c, self.d = pa, pb, pc, pd
            self.world_x = self.x * pa + self.y * pb + px
            self.world_y = self.x * pc + self.y * pd + py
        elif tm == TM_NO_ROT_TRANS:
            # noRotationOrReflection:丢掉父级旋转和反射,只留父级的缩放/切变
            # 位置仍用原始父矩阵算,下面 pa/pb/pc/pd 会被改写,所以先算
            self.world_x = self.x * pa + self.y * pb + px
            self.world_y = self.x * pc + self.y * pd + py
            s = pa * pa + pc * pc
            if s > 0.0001:
                s = abs(pa * pd - pb * pc) / s
                pb = pc * s
                pd = pa * s
                prx = math.atan2(pc, pa) / D2R
            else:
                pa = 0.0
                pc = 0.0
                prx = 90.0 - math.atan2(pd, pb) / D2R
            rx = self.rotation + self.shear_x - prx
            ry = self.rotation + self.shear_y - prx + 90.0
            ra = math.cos(rx * D2R) * self.scale_x
            rb = math.cos(ry * D2R) * self.scale_y
            rc = math.sin(rx * D2R) * self.scale_x
            rd = math.sin(ry * D2R) * self.scale_y
            self.a = pa * ra - pb * rc
            self.b = pa * rb - pb * rd
            self.c = pc * ra + pd * rc
            self.d = pc * rb + pd * rd
        else:  # TM_NO_SCALE_REFL
            s = math.sqrt(abs(pa * pa + pb * pb)) or 1.0
            self.world_x = self.x * pa + self.y * pb + px
            self.world_y = self.x * pc + self.y * pd + py
            self.a = (pa * la + pb * lc) / s
            self.b = (pa * lb + pb * ld) / s
            self.c = (pc * la + pd * lc) / s
            self.d = (pc * lb + pd * ld) / s

        self.world_rotation = math.atan2(self.c, self.a) / D2R
        self.world_scale_x = math.sqrt(self.a * self.a + self.c * self.c)
        self.world_scale_y = math.sqrt(self.b * self.b + self.d * self.d)

    def world_to_local(self, wx, wy):
        det = self.a * self.d - self.b * self.c
        if det == 0:
            return 0.0, 0.0
        dx = wx - self.world_x
        dy = wy - self.world_y
        return (dx * self.d - dy * self.b) / det, (dy * self.a - dx * self.c) / det


# ============================ 附件顶点计算 ============================

def _region_offset(att, region=None):
    """区域附件的四个本地顶点(左下/右下/右上/左上),spine 顺序"""
    w = float(att.get('width', 0.0) or 0.0)
    h = float(att.get('height', 0.0) or 0.0)
    lx1, lx2 = -w / 2.0, w / 2.0
    ly1, ly2 = -h / 2.0, h / 2.0
    if region is not None and region.orig_w and region.orig_h:
        # 图集裁掉了空白,绘制范围要收窄到裁剪后的子矩形
        # 图集里的尺寸是纹理像素,可能和骨架单位不是 1:1(图集打包缩放),要按比例换算
        kx = w / float(region.orig_w)
        ky = h / float(region.orig_h)
        lx1 = -w / 2.0 + region.off_x * kx
        ly1 = -h / 2.0 + region.off_y * ky
        lx2 = lx1 + region.w * kx
        ly2 = ly1 + region.h * ky
    lxs = (lx1, lx1, lx2, lx2)
    lys = (ly1, ly2, ly2, ly1)
    sx = float(att.get('scaleX', 1.0) or 1.0)
    sy = float(att.get('scaleY', 1.0) or 1.0)
    rot = float(att.get('rotation', 0.0) or 0.0)
    ox = float(att.get('x', 0.0) or 0.0)
    oy = float(att.get('y', 0.0) or 0.0)
    cos = math.cos(rot * D2R)
    sin = math.sin(rot * D2R)
    out = []
    for px, py in zip(lxs, lys):
        px *= sx
        py *= sy
        out.append(px * cos - py * sin + ox)
        out.append(px * sin + py * cos + oy)
    return out


def _region_uvs(att, region, page_w, page_h):
    """区域附件四个顶点的 uv(左下/左上/右上/右下),顺序与顶点一致"""
    u = region.x / page_w
    v = region.y / page_h
    if region.rotate:
        # 图集里这块是转了90°存的,贴图里的实际宽高要对调
        u2 = (region.x + region.h) / page_w
        v2 = (region.y + region.w) / page_h
        return [(u2, v2), (u, v2), (u, v), (u2, v)]
    u2 = (region.x + region.w) / page_w
    v2 = (region.y + region.h) / page_h
    return [(u, v2), (u, v), (u2, v), (u2, v2)]


def _is_weighted(verts, vertex_count):
    if not verts:
        return False
    if isinstance(verts[0], (list, tuple)):
        return True
    return len(verts) != vertex_count * 2


def mesh_world_vertices(sk, att, bone, deform=None):
    """网格顶点的世界坐标

    加权网格(spine 里每个顶点带 bones/weights)每个顶点要按各骨骼权重混合,
    当成"单根骨的局部坐标"来画就会散架,这是之前动画播放乱掉的主因。
    """
    bones_arr = att.get('bones')
    if not bones_arr:
        vertex_count = int(att.get('vertexCount') or 0)
        if vertex_count <= 0:
            vertex_count = len(att.get('uvs') or []) // 2
        ev = None
        if deform:
            ev = [(deform[i * 2] if i * 2 < len(deform) else 0.0,
                   deform[i * 2 + 1] if i * 2 + 1 < len(deform) else 0.0)
                  for i in range(vertex_count)]
        return [(lx * bone.a + ly * bone.b + bone.world_x,
                 lx * bone.c + ly * bone.d + bone.world_y)
                for lx, ly in mesh_vertices_local(att, ev)]

    verts = att.get('vertices') or []
    all_bones = sk.bones
    out = []
    v = b = f = 0
    while v < len(bones_arr):
        n = int(bones_arr[v])
        v += 1
        wx = wy = 0.0
        for _ in range(n):
            bi = int(bones_arr[v])
            v += 1
            vx = verts[b]
            vy = verts[b + 1]
            w = verts[b + 2]
            if deform:
                # deform 的偏移量是按"权重项"编号的,不是按顶点
                if f < len(deform):
                    vx += deform[f]
                if f + 1 < len(deform):
                    vy += deform[f + 1]
            bm = all_bones[bi]
            wx += (vx * bm.a + vy * bm.b + bm.world_x) * w
            wy += (vx * bm.c + vy * bm.d + bm.world_y) * w
            b += 3
            f += 2
        out.append((wx, wy))
    return out


def mesh_vertices_local(att, edit_vertices=None):
    """返回网格顶点本地坐标列表 [(x,y), ...](已应用 deform)"""
    vertex_count = int(att.get('vertexCount') or 0)
    verts = att.get('vertices') or []
    if vertex_count <= 0:
        # JSON 不带 vertexCount,顶点数等于 uv 数量的一半
        vertex_count = len(att.get('uvs') or []) // 2
    out = []
    if isinstance(verts[0], (list, tuple)) if verts else False:
        # 官方 JSON 的嵌套加权格式
        for item in verts:
            bone_count = int(item[0])
            bx = by = 0.0
            for bi in range(bone_count):
                base = 1 + bi * 4
                # 加权网格的绝对坐标(不含骨骼变换),这里只取权重最大项近似
                bx += float(item[base + 1]) * float(item[base + 3])
                by += float(item[base + 2]) * float(item[base + 3])
            out.append((bx, by))
        return out

    if not _is_weighted(verts, vertex_count):
        for i in range(vertex_count):
            x = verts[i * 2]
            y = verts[i * 2 + 1]
            if edit_vertices is not None and i < len(edit_vertices):
                x += edit_vertices[i][0]
                y += edit_vertices[i][1]
            out.append((x, y))
        return out

    # 二进制解析出的扁平加权格式: [boneCount, (boneIdx,x,y,w)*boneCount, ...]
    i = 0
    vi = 0
    while i < len(verts) and len(out) < vertex_count:
        bone_count = int(verts[i])
        i += 1
        bx = by = 0.0
        for _ in range(bone_count):
            i += 1  # bone index
            x = verts[i]
            y = verts[i + 1]
            w = verts[i + 2]
            i += 3
            bx += x * w
            by += y * w
        if edit_vertices is not None and vi < len(edit_vertices):
            bx += edit_vertices[vi][0]
            by += edit_vertices[vi][1]
        out.append((bx, by))
        vi += 1
    return out


# ============================ 骨架 ============================

class Skeleton:
    def __init__(self, data):
        self.data = data
        sk = data.get('skeleton', {}) or {}
        self.width = float(sk.get('width', 0.0) or 0.0)
        self.height = float(sk.get('height', 0.0) or 0.0)
        self.version = sk.get('spine', '') or ''
        self.fps = float(sk.get('fps', 30.0) or 30.0)

        self.bones = []
        self.bones_by_name = {}
        for bd in data.get('bones', []) or []:
            parent = None
            pn = bd.get('parent')
            if pn:
                parent = self.bones_by_name.get(pn)
            bone = Bone(bd, parent)
            self.bones.append(bone)
            self.bones_by_name[bone.name] = bone

        self.slots = []
        self.slots_by_name = {}
        for sd in data.get('slots', []) or []:
            slot = {
                'name': sd.get('name'),
                'bone': self.bones_by_name.get(sd.get('bone')),
                'attachment': sd.get('attachment'),
                'color': _hex_to_rgba(sd.get('color', 'ffffffff')),
                'dark': _hex_to_rgba(sd.get('dark', 'ffffffff')),
                'blend': _BLEND_MODE.get(sd.get('blend', 'normal'), BLEND_NORMAL),
            }
            self.slots.append(slot)
            self.slots_by_name[slot['name']] = slot

        # 皮肤: {skinName: {slotName: {attName: att}}}
        self.skins = {}
        skins = data.get('skins')
        if isinstance(skins, list):
            for s in skins:
                self.skins[s.get('name', 'default')] = s.get('attachments', {}) or {}
        elif isinstance(skins, dict):
            self.skins = skins
        self.default_skin = 'default' if 'default' in self.skins else (
            next(iter(self.skins)) if self.skins else None)
        self.skin = self.skins.get(self.default_skin, {}) if self.default_skin else {}

        # JSON 里 region 不带 type,附件名只存在于皮肤字典的 key 上,这里补全
        for skin in self.skins.values():
            for atts in skin.values():
                for att_name, att in atts.items():
                    if not isinstance(att, dict):
                        continue
                    if not att.get('name'):
                        att['name'] = att_name
                    if not att.get('type'):
                        att['type'] = 'region'
                    flat = att.get('vertices') or []
                    # 顶点数比 uv 多说明是加权网格:扁平结构 [骨数, (骨序号,x,y,权重)*n],
                    # 拆成 bones/weights 两组,画的时候才好按权重混合
                    if (att.get('type') == 'mesh' and flat
                            and len(flat) != len(att.get('uvs') or [])
                            and not att.get('bones')):
                        bones = []
                        weights = []
                        i = 0
                        while i < len(flat):
                            cnt = int(flat[i]); i += 1
                            bones.append(cnt)
                            for _ in range(cnt):
                                bones.append(int(flat[i]))
                                weights.append(flat[i + 1])
                                weights.append(flat[i + 2])
                                weights.append(flat[i + 3])
                                i += 4
                        att['bones'] = bones
                        att['vertices'] = weights

        self.ik = []
        for ic in data.get('ik', []) or []:
            bones = [self.bones_by_name.get(n) for n in ic.get('bones', [])]
            bones = [b for b in bones if b]
            if not bones:
                continue
            self.ik.append({
                'name': ic.get('name'), 'order': int(ic.get('order', 0) or 0),
                'bones': bones, 'target': self.bones_by_name.get(ic.get('target')),
                'mix': float(ic.get('mix', 1.0) or 1.0),
                'bend_positive': bool(ic.get('bendPositive', True)),
                'compress': bool(ic.get('compress', False)),
                'stretch': bool(ic.get('stretch', False)),
                'uniform': bool(ic.get('uniform', False)),
            })

        self.transform = []
        for tc in data.get('transform', []) or []:
            bone = self.bones_by_name.get((tc.get('bones') or [None])[0])
            if bone is None:
                continue
            self.transform.append({
                'name': tc.get('name'), 'order': int(tc.get('order', 0) or 0),
                'bone': bone, 'target': self.bones_by_name.get(tc.get('target')),
                'local': bool(tc.get('local', False)),
                'relative': bool(tc.get('relative', False)),
                'offset_rotation': float(tc.get('rotation', 0.0) or 0.0),
                'offset_x': float(tc.get('x', 0.0) or 0.0),
                'offset_y': float(tc.get('y', 0.0) or 0.0),
                'offset_scale_x': float(tc.get('scaleX', 0.0) or 0.0),
                'offset_scale_y': float(tc.get('scaleY', 0.0) or 0.0),
                'offset_shear_y': float(tc.get('shearY', 0.0) or 0.0),
                'rotate_mix': float(tc.get('rotateMix', 1.0) or 1.0),
                'translate_mix': float(tc.get('translateMix', 1.0) or 1.0),
                'scale_mix': float(tc.get('scaleMix', 1.0) or 1.0),
                'shear_mix': float(tc.get('shearMix', 1.0) or 1.0),
            })

        self.path = []
        for pc in data.get('path', []) or []:
            bones = [self.bones_by_name.get(n) for n in pc.get('bones', [])]
            bones = [b for b in bones if b]
            if not bones:
                continue
            self.path.append({
                'name': pc.get('name'), 'order': int(pc.get('order', 0) or 0),
                'bones': bones,
                'target': self.slots_by_name.get(pc.get('target')),
                'position_mode': pc.get('positionMode', 'percent'),
                'spacing_mode': pc.get('spacingMode', 'length'),
                'rotate_mode': pc.get('rotateMode', 'tangent'),
                'rotation': float(pc.get('rotation', 0.0) or 0.0),
                'position': float(pc.get('position', 0.0) or 0.0),
                'spacing': float(pc.get('spacing', 0.0) or 0.0),
                'rotate_mix': float(pc.get('rotateMix', 1.0) or 1.0),
                'translate_mix': float(pc.get('translateMix', 1.0) or 1.0),
            })

        self.animations = data.get('animations', {}) or {}
        self.atlas = None       # 由 renderer 注入
        self.textures = {}      # page -> np.array

        self.animation = None
        self.animation_name = None
        self.time = 0.0
        self.duration = 0.0
        self.loop = True
        self._draw_order = list(range(len(self.slots)))
        self.apply_setup()

    # ---------- 姿态 ----------
    def apply_setup(self):
        for b in self.bones:
            b.reset()
        for s in self.slots:
            s['color'] = _hex_to_rgba(s.get('_color_hex', None)) if s.get('_color_hex') else s['color']

    def reset_pose(self):
        for b in self.bones:
            b.reset()
        # 槽位颜色回到配置
        for sd, s in zip(self.data.get('slots', []) or [], self.slots):
            s['color'] = _hex_to_rgba(sd.get('color', 'ffffffff'))
            if sd.get('attachment'):
                s['attachment'] = sd.get('attachment')

    def set_skin(self, name):
        if name in self.skins:
            self.default_skin = name
            self.skin = self.skins[name]

    def set_animation(self, name):
        self.animation_name = name
        self.animation = self.animations.get(name)
        self.time = 0.0
        self.duration = self._duration(self.animation) if self.animation else 0.0

    @staticmethod
    def _duration(anim):
        if not anim:
            return 0.0

        def walk(obj):
            best = 0.0
            if isinstance(obj, list):
                for f in obj:
                    if isinstance(f, dict):
                        t = f.get('time')
                        if t is not None:
                            best = max(best, t)
                    else:
                        best = max(best, walk(f))
            elif isinstance(obj, dict):
                for v in obj.values():
                    best = max(best, walk(v))
            return best

        # 事件/绘制顺序不是必要数据,但 time 同样有效,一起扫即可
        return walk(anim)

    def update(self, dt):
        if self.animation is None or self.duration <= 0:
            self.time = 0.0
        else:
            self.time += dt
            if self.loop:
                if self.time > self.duration:
                    self.time = self.time % self.duration if self.duration > 0 else 0.0
            else:
                self.time = min(self.time, self.duration)
        self.apply_pose(self.time)

    # ---------- 应用动画 ----------
    def apply_pose(self, time):
        # 网格变形量是按当前时间取的,这里不同步的话播放时变形一直停在旧帧
        self.time = time
        self.reset_pose()
        anim = self.animation
        self._draw_order = list(range(len(self.slots)))

        if anim:
            # 槽位时间轴
            for slot_name, tl in (anim.get('slots') or {}).items():
                slot = self.slots_by_name.get(slot_name)
                if not slot:
                    continue
                att = tl.get('attachment')
                if att:
                    i, _ = _find_frames(att, time)
                    slot['attachment'] = att[i].get('name')
                col = tl.get('color')
                if col:
                    v = sample(col, time, ['color'], {'color': 'ffffffff'})
                    slot['color'] = _hex_to_rgba(v['color'])
                tw = tl.get('twoColor')
                if tw:
                    v = sample(tw, time, ['light', 'dark'], {'light': 'ffffffff', 'dark': 'ffffffff'})
                    slot['color'] = _hex_to_rgba(v['light'])

            # 骨骼时间轴
            for bone_name, tl in (anim.get('bones') or {}).items():
                bone = self.bones_by_name.get(bone_name)
                if not bone:
                    continue
                rot = tl.get('rotate')
                if rot:
                    v = sample(rot, time, ['angle'], {'angle': 0.0})
                    bone.rotation = bone.setup[0] + v['angle']
                tr = tl.get('translate')
                if tr:
                    v = sample(tr, time, ['x', 'y'], {'x': 0.0, 'y': 0.0})
                    bone.x = bone.setup[1] + v['x']
                    bone.y = bone.setup[2] + v['y']
                sc = tl.get('scale')
                if sc:
                    v = sample(sc, time, ['x', 'y'], {'x': 1.0, 'y': 1.0})
                    bone.scale_x = bone.setup[3] * v['x']
                    bone.scale_y = bone.setup[4] * v['y']
                sh = tl.get('shear')
                if sh:
                    v = sample(sh, time, ['x', 'y'], {'x': 0.0, 'y': 0.0})
                    bone.shear_x = bone.setup[5] + v['x']
                    bone.shear_y = bone.setup[6] + v['y']

            # IK 时间轴
            for name, tl in (anim.get('ik') or {}).items():
                for c in self.ik:
                    if c['name'] == name:
                        v = sample(tl, time, ['mix', 'bendPositive'], {'mix': 1.0, 'bendPositive': True})
                        c['mix'] = v['mix']
                        c['bend_positive'] = v['bendPositive'] >= 0.5

            # Transform 时间轴
            for name, tl in (anim.get('transform') or {}).items():
                for c in self.transform:
                    if c['name'] == name:
                        v = sample(tl, time, ['rotateMix', 'translateMix', 'scaleMix', 'shearMix'],
                                   {'rotateMix': 1.0, 'translateMix': 1.0, 'scaleMix': 1.0, 'shearMix': 1.0})
                        c['rotate_mix'] = v['rotateMix']
                        c['translate_mix'] = v['translateMix']
                        c['scale_mix'] = v['scaleMix']
                        c['shear_mix'] = v['shearMix']

            # 路径时间轴
            for name, tl in (anim.get('paths') or {}).items():
                for c in self.path:
                    if c['name'] == name:
                        if 'position' in tl:
                            v = sample(tl['position'], time, ['position'], {'position': 0.0})
                            c['position'] = v['position']
                        if 'spacing' in tl:
                            v = sample(tl['spacing'], time, ['spacing'], {'spacing': 0.0})
                            c['spacing'] = v['spacing']
                        if 'mix' in tl:
                            v = sample(tl['mix'], time, ['mix'], {'mix': 1.0})
                            c['rotate_mix'] = c['translate_mix'] = v['mix']

            # 绘制顺序时间轴
            do = anim.get('drawOrder')
            if do:
                i, _ = _find_frames(do, time)
                for entry in do[i].get('offsets', []) or []:
                    sname = entry.get('slot')
                    off = int(entry.get('offset', 0) or 0)
                    if off == 0 or sname not in self.slots_by_name:
                        continue
                    si = self.slots.index(self.slots_by_name[sname])
                    try:
                        cur = self._draw_order.index(si)
                    except ValueError:
                        continue
                    self._draw_order.pop(cur)
                    self._draw_order.insert(max(0, min(len(self._draw_order), cur + off)), si)

        # 世界变换
        for b in self.bones:
            b.update()
        self._apply_constraints()

    def _apply_constraints(self):
        for c in sorted(self.ik, key=lambda x: x['order']):
            try:
                _solve_ik(c)
                # 父骨转完,它下面挂的旁支也要跟着重算,不能只刷 IK 链的末端
                self._update_subtree(c['bones'][0])
            except Exception:
                pass
        for c in sorted(self.transform, key=lambda x: x['order']):
            try:
                _solve_transform(c)
                self._update_subtree(c['bone'])
            except Exception:
                pass
        for c in sorted(self.path, key=lambda x: x['order']):
            try:
                _solve_path(c, self)
                if c['bones']:
                    self._update_subtree(c['bones'][-1])
            except Exception:
                pass

    def _update_subtree(self, bone):
        stack = list(bone.children)
        while stack:
            b = stack.pop()
            b.update()
            stack.extend(b.children)

    # ---------- 附件查询 ----------
    def get_attachment(self, slot):
        att_name = slot.get('attachment')
        if not att_name:
            return None
        slot_atts = self.skin.get(slot['name']) if self.skin else None
        att = None
        if slot_atts:
            att = slot_atts.get(att_name)
        if att is None:
            for sname, skin in self.skins.items():
                if slot['name'] in skin and att_name in skin[slot['name']]:
                    att = skin[slot['name']][att_name]
                    break
        return att

    def deform_offsets(self, skin_name, slot_name, att_name, time):
        """取 deform 时间轴的顶点偏移"""
        anim = self.animation
        if not anim:
            return None
        deform = anim.get('deform') or {}
        skin = deform.get(skin_name)
        if not skin:
            return None
        slot = skin.get(slot_name)
        if not slot:
            return None
        frames = slot.get(att_name)
        if not frames:
            return None
        i, between = _find_frames(frames, time)
        f = frames[i]

        def build(fr):
            if not fr.get('vertices'):
                return None
            off = int(fr.get('offset', 0) or 0)
            return (off, fr['vertices'])

        base = build(f)
        if not between:
            return base[1] if base else None
        nxt = build(frames[i + 1])
        if base is None and nxt is None:
            return None
        if base is None:
            base = nxt
        if nxt is None:
            nxt = base
        pct = _curve_pct(f, (time - _ftime(f)) / (_ftime(frames[i + 1]) - _ftime(f))
                         if _ftime(frames[i + 1]) > _ftime(f) else 0.0)
        n = max(len(base[1]), len(nxt[1]))
        bv = list(base[1]) + [0.0] * (n - len(base[1]))
        nv = list(nxt[1]) + [0.0] * (n - len(nxt[1]))
        # 取较小 offset 为基准
        off = min(base[0], nxt[0])
        pad_b = base[0] - off
        pad_n = nxt[0] - off
        size = max(pad_b + len(bv), pad_n + len(nv))
        out = [0.0] * size
        for k in range(len(bv)):
            out[pad_b + k] += bv[k] * (1 - pct)
        for k in range(len(nv)):
            out[pad_n + k] += nv[k] * pct
        return out


def _hex_to_rgba(h):
    if not h:
        return (1.0, 1.0, 1.0, 1.0)
    h = str(h).strip()
    if len(h) == 6:
        h += 'ff'
    if len(h) < 8:
        h = h.ljust(8, 'f')
    try:
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0,
                int(h[4:6], 16) / 255.0, int(h[6:8], 16) / 255.0)
    except Exception:
        return (1.0, 1.0, 1.0, 1.0)


# ============================ 约束求解 ============================

def _set_local(bone, x, y, rotation, scale_x, scale_y, shear_x, shear_y):
    """约束求解里要直接改骨骼局部变换,改完立刻刷新世界矩阵"""
    bone.x = x
    bone.y = y
    bone.rotation = rotation
    bone.scale_x = scale_x
    bone.scale_y = scale_y
    bone.shear_x = shear_x
    bone.shear_y = shear_y
    bone.update()


def _ik_apply1(bone, target_x, target_y, compress, stretch, uniform, alpha):
    """单骨骼 IK:只转这一根骨,让它末端尽量贴住目标"""
    parent = bone.parent
    if parent is None:
        return
    mode = bone.transform_mode
    tx = ty = 0.0
    rotation_ik = -bone.shear_x - bone.rotation
    if mode == TM_ONLY_TRANSLATION:
        tx = target_x - bone.world_x
        ty = target_y - bone.world_y
    elif mode == TM_NO_ROT_TRANS:
        # 这里目标是相对"去掉父级旋转"的坐标系,和 Bone.update 那套保持一致
        pa, pb, pc, pd = parent.a, parent.b, parent.c, parent.d
        s = pa * pa + pc * pc
        if s > 0.0001:
            s = abs(pa * pd - pb * pc) / s
            pb = -pc * s
            pd = pa * s
            rotation_ik += math.atan2(pc, pa) / D2R
        det = pa * pd - pb * pc
        if abs(det) > 0.000001:
            dx = target_x - parent.world_x
            dy = target_y - parent.world_y
            tx = (dx * pd - dy * pb) / det - bone.x
            ty = (dy * pa - dx * pc) / det - bone.y
    else:
        lx, ly = parent.world_to_local(target_x, target_y)
        tx = lx - bone.x
        ty = ly - bone.y
    rotation_ik += math.atan2(ty, tx) / D2R
    if bone.scale_x < 0:
        rotation_ik += 180.0
    if rotation_ik > 180.0:
        rotation_ik -= 360.0
    elif rotation_ik < -180.0:
        rotation_ik += 360.0
    sx, sy = bone.scale_x, bone.scale_y
    if compress or stretch:
        if mode == TM_NO_SCALE_REFL:
            tx = target_x - bone.world_x
            ty = target_y - bone.world_y
        blen = bone.length * sx
        dd = math.sqrt(tx * tx + ty * ty)
        if blen > 0.0001 and ((compress and dd < blen) or (stretch and dd > blen)):
            s = (dd / blen - 1) * alpha + 1
            sx *= s
            if uniform:
                sy *= s
    # 官方是"算出目标角度后整体赋值",累加会越算越偏
    _set_local(bone, bone.x, bone.y, bone.rotation + rotation_ik * alpha,
               sx, sy, bone.shear_x, bone.shear_y)


def _ik_apply2(parent, child, target_x, target_y, bend_dir, stretch, alpha):
    """双骨骼 IK:解出父子两根骨的旋转,让子骨末端对准目标"""
    g = parent.parent
    if g is None:
        return
    px, py = parent.x, parent.y
    psx = sx = parent.scale_x
    psy = parent.scale_y
    csx = child.scale_x
    s2 = 1.0
    os1 = 0.0
    if psx < 0:
        psx = -psx
        os1 = 180.0
        s2 = -1.0
    if psy < 0:
        psy = -psy
        s2 = -s2
    os2 = 0.0
    if csx < 0:
        csx = -csx
        os2 = 180.0

    cx = child.x
    uniform = abs(psx - psy) <= 0.0001
    if uniform:
        cy = child.y
        cwx = parent.a * cx + parent.b * cy + parent.world_x
        cwy = parent.c * cx + parent.d * cy + parent.world_y
    else:
        # 非等比缩放时子骨的 y 偏移会被父级切变吃掉,官方这里直接按 0 处理
        cy = 0.0
        cwx = parent.a * cx + parent.world_x
        cwy = parent.c * cx + parent.world_y

    gx, gy = g.world_to_local(cwx, cwy)
    dx, dy = gx - px, gy - py
    l1 = math.sqrt(dx * dx + dy * dy)
    l2 = child.length * csx
    if l1 < 0.0001:
        _ik_apply1(parent, target_x, target_y, False, stretch, False, alpha)
        _set_local(child, cx, cy, 0.0, child.scale_x, child.scale_y,
                   child.shear_x, child.shear_y)
        return

    tx, ty = g.world_to_local(target_x, target_y)
    tx -= px
    ty -= py
    dd = tx * tx + ty * ty

    if uniform:
        l2 *= psx
        cos = (dd - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if cos < -1.0:
            cos = -1.0
        elif cos > 1.0:
            cos = 1.0
            if stretch:
                sx *= (math.sqrt(dd) / (l1 + l2) - 1) * alpha + 1
        a2 = math.acos(cos) * bend_dir
        ca = l1 + l2 * cos
        cb = l2 * math.sin(a2)
        a1 = math.atan2(ty * ca - tx * cb, tx * ca + ty * cb)
    else:
        a = psx * l2
        b = psy * l2
        aa = a * a
        bb = b * b
        ta = math.atan2(ty, tx)
        cc = bb * l1 * l1 + aa * dd - aa * bb
        c1 = -2 * bb * l1
        c2 = bb - aa
        disc = c1 * c1 - 4 * c2 * cc
        a1 = a2 = 0.0
        solved = False
        if disc >= 0:
            q = math.sqrt(disc)
            if c1 < 0:
                q = -q
            q = -(c1 + q) / 2
            if q != 0:
                r0 = q / c2
                r1 = cc / q
                r = r0 if abs(r0) < abs(r1) else r1
                if r * r <= dd:
                    yy = math.sqrt(dd - r * r) * bend_dir
                    a1 = ta - math.atan2(yy, r)
                    a2 = math.atan2(yy / psy, (r - l1) / psx)
                    solved = True
        if not solved:
            # 解不唯一时在可达范围里挑离目标最近的角度
            min_angle, min_x, min_dist, min_y = math.pi, l1 - a, (l1 - a) ** 2, 0.0
            max_angle, max_x, max_dist, max_y = 0.0, l1 + a, (l1 + a) ** 2, 0.0
            t = -a * l1 / (aa - bb)
            if -1.0 <= t <= 1.0:
                t = math.acos(t)
                xx = a * math.cos(t) + l1
                yy = b * math.sin(t)
                d = xx * xx + yy * yy
                if d < min_dist:
                    min_angle, min_x, min_y, min_dist = t, xx, yy, d
                if d > max_dist:
                    max_angle, max_x, max_y, max_dist = t, xx, yy, d
            if dd <= (min_dist + max_dist) / 2:
                a1 = ta - math.atan2(min_y * bend_dir, min_x)
                a2 = min_angle * bend_dir
            else:
                a1 = ta - math.atan2(max_y * bend_dir, max_x)
                a2 = max_angle * bend_dir

    os = math.atan2(cy, cx) * s2
    rot = parent.rotation
    ra1 = (a1 - os) / D2R + os1 - rot
    if ra1 > 180.0:
        ra1 -= 360.0
    elif ra1 < -180.0:
        ra1 += 360.0
    # 父骨的切变被抹掉是官方的做法,别"修"
    _set_local(parent, px, py, rot + ra1 * alpha, sx, parent.scale_y, 0.0, 0.0)
    rot = child.rotation
    ra2 = ((a2 + os) / D2R - child.shear_x) * s2 + os2 - rot
    if ra2 > 180.0:
        ra2 -= 360.0
    elif ra2 < -180.0:
        ra2 += 360.0
    _set_local(child, cx, cy, rot + ra2 * alpha, child.scale_x, child.scale_y,
               child.shear_x, child.shear_y)


def _solve_ik(c):
    bones = c['bones']
    target = c['target']
    if target is None:
        return
    mix = c['mix']
    if mix == 0:
        return
    if len(bones) == 1:
        _ik_apply1(bones[0], target.world_x, target.world_y,
                   c['compress'], c['stretch'], c['uniform'], mix)
    elif len(bones) >= 2:
        _ik_apply2(bones[0], bones[1], target.world_x, target.world_y,
                   1.0 if c['bend_positive'] else -1.0, c['stretch'], mix)


def _solve_transform(c):
    bone = c['bone']
    target = c['target']
    if target is None:
        return
    # 目标到本骨父空间的局部值
    if c['local']:
        tr = target.rotation
        tx, ty = target.x, target.y
        tsx, tsy = target.scale_x, target.scale_y
        tshy = target.shear_y
    else:
        tr = target.world_rotation
        tx, ty = target.world_x, target.world_y
        tsx, tsy = target.world_scale_x, target.world_scale_y
        # 再转成相对本骨父空间
        p = bone.parent
        if p is not None and not c['relative']:
            lx, ly = p.world_to_local(tx, ty)
            tx, ty = lx, ly
            tr = tr - p.world_rotation

    if c['relative']:
        tr -= target.parent.world_rotation if (target.parent and not c['local']) else 0.0

    if c['rotate_mix'] > 0:
        rotate = tr + c['offset_rotation']
        bone.rotation = bone.rotation + (rotate - bone.rotation) * c['rotate_mix']
    if c['translate_mix'] > 0:
        tx += c['offset_x']
        ty += c['offset_y']
        bone.x = bone.x + (tx - bone.x) * c['translate_mix']
        bone.y = bone.y + (ty - bone.y) * c['translate_mix']
    if c['scale_mix'] > 0:
        sx = tsx + c['offset_scale_x']
        sy = tsy + c['offset_scale_y']
        if target.scale_x < 0:
            sx = -sx
        bone.scale_x = bone.scale_x + (sx - bone.scale_x) * c['scale_mix']
        bone.scale_y = bone.scale_y + (sy - bone.scale_y) * c['scale_mix']
    if c['shear_mix'] > 0:
        shy = tshy + c['offset_shear_y']
        bone.shear_y = bone.shear_y + (shy - bone.shear_y) * c['shear_mix']


def _solve_path(c, skeleton):
    slot = c['target']
    if slot is None:
        return
    att = skeleton.get_attachment(slot)
    if not att or att.get('type') != 'path':
        return
    verts = att.get('vertices') or []
    lengths = att.get('lengths') or []
    if len(verts) < 6:
        return
    pts = [(verts[i], verts[i + 1]) for i in range(0, len(verts), 2)]
    total = sum(lengths) if lengths else 1.0
    if total <= 0:
        return
    bones = c['bones']
    spacing = c['spacing']
    pos = c['position']
    if c['position_mode'] == 'percent':
        pos *= total
    if c['spacing_mode'] == 'percent':
        spacing *= total
    scale = abs(bones[0].scale_x) or 1.0
    if c['spacing_mode'] == 'fixed':
        spacing *= scale

    for idx, bone in enumerate(bones):
        d = pos + spacing * idx
        if c['spacing_mode'] in ('percent', 'length'):
            d = (pos + spacing * idx)
        # 找落在哪一段
        acc = 0.0
        px = py = 0.0
        tangent = 0.0
        for seg in range(len(lengths)):
            seg_len = lengths[seg]
            if d <= acc + seg_len or seg == len(lengths) - 1:
                t = 0.0 if seg_len <= 0 else (d - acc) / seg_len
                x1, y1 = pts[seg]
                x2, y2 = pts[seg + 1]
                px = x1 + (x2 - x1) * t
                py = y1 + (y2 - y1) * t
                tangent = math.atan2(y2 - y1, x2 - x1) / D2R
                break
            acc += seg_len
        if c['translate_mix'] > 0:
            bone.x = bone.x + (px - bone.x) * c['translate_mix']
            bone.y = bone.y + (py - bone.y) * c['translate_mix']
        if c['rotate_mix'] > 0:
            rot = tangent + c['rotation']
            bone.rotation = bone.rotation + (rot - bone.rotation) * c['rotate_mix']


# ============================ 光栅化 ============================

def _raster_tri(dst, tex, p0, p1, p2, tint, blend):
    """dst: (H,W,4) float RGBA; tex: (h,w,4) float; p=(x,y,u,v) 像素坐标 + uv[0..1]"""
    H, W = dst.shape[0], dst.shape[1]
    tex_h, tex_w = tex.shape[0], tex.shape[1]
    x0, y0, u0, v0 = p0
    x1, y1, u1, v1 = p1
    x2, y2, u2, v2 = p2

    minx = int(math.floor(min(x0, x1, x2)))
    maxx = int(math.ceil(max(x0, x1, x2)))
    miny = int(math.floor(min(y0, y1, y2)))
    maxy = int(math.ceil(max(y0, y1, y2)))
    minx = max(minx, 0)
    miny = max(miny, 0)
    maxx = min(maxx, W - 1)
    maxy = min(maxy, H - 1)
    if maxx < minx or maxy < miny:
        return

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-9:
        return

    ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    l0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
    l1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
    l2 = 1.0 - l0 - l1
    mask = (l0 >= -0.0005) & (l1 >= -0.0005) & (l2 >= -0.0005)
    if not mask.any():
        return

    u = l0 * u0 + l1 * u1 + l2 * u2
    v = l0 * v0 + l1 * v1 + l2 * v2
    tx = np.clip((u * (tex_w - 1)).astype(np.int32), 0, tex_w - 1)
    ty = np.clip((v * (tex_h - 1)).astype(np.int32), 0, tex_h - 1)
    src = tex[ty, tx]
    a = np.where(mask[..., None], src[..., 3:4] * tint[3], 0.0)
    if blend == BLEND_NORMAL:
        rgb = src[..., :3] * tint[:3]
    else:
        rgb = src[..., :3] * tint[:3]
    region = dst[miny:maxy + 1, minx:maxx + 1]
    if blend == BLEND_ADDITIVE:
        region[..., :3] += rgb * a
    else:
        region[..., :3] = rgb * a + region[..., :3] * (1.0 - a)
    region[..., 3:4] = a + region[..., 3:4] * (1.0 - a)


class Renderer:
    def __init__(self):
        self.skeleton = None
        self.regions = {}
        self.pages = {}
        self.page_images = {}
        self.page_arrays = {}
        self.state = 'none'   # none / ok / error
        self.error = ''

    # ---------- 加载 ----------
    def load(self, path):
        """path 可以是骨架文件(.json/.skel 转换后的产物)或所在目录"""
        directory, skel_path, atlas_path = _locate(path)
        if skel_path is None:
            self._fail('未找到骨架文件(.json/.skel)')
            return False
        if atlas_path is None:
            self._fail('未找到 .atlas 文件')
            return False
        try:
            with open(skel_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self._fail(f'骨架JSON解析失败: {e}')
            return False
        return self.load_from_data(data, atlas_path)

    def load_from_data(self, data, atlas_path):
        """用已解析好的骨架字典 + 图集文件构建,供 .skel 转换后的数据复用"""
        self.skeleton = None
        self.regions = {}
        self.pages = {}
        self.page_images = {}
        self.page_arrays = {}
        self.error = ''

        try:
            self.regions, self.pages = load_atlas(atlas_path)
        except Exception:
            self.regions = {}
            self.pages = {}

        # 加载贴图
        atlas_dir = os.path.dirname(atlas_path)
        for page_name in set(self.pages.values()):
            img_path = os.path.join(atlas_dir, page_name)
            if not os.path.isfile(img_path):
                # 兼容大小写/子目录
                found = _find_file(atlas_dir, page_name)
                if found:
                    img_path = found
                else:
                    continue
            try:
                img = Image.open(img_path).convert('RGBA')
                self.page_images[page_name] = img
                self.page_arrays[page_name] = np.asarray(img, dtype=np.float32) / 255.0
            except Exception:
                continue

        try:
            self.skeleton = Skeleton(data)
            self.skeleton.atlas = (self.regions, self.pages)
            self.skeleton.textures = self.page_arrays
        except Exception as e:
            self._fail(f'骨架构建失败: {e}')
            return False

        self.state = 'ok'
        return True

    def _fail(self, msg):
        self.state = 'error'
        self.error = msg

    @property
    def animation_names(self):
        if not self.skeleton:
            return []
        return list(self.skeleton.animations.keys())

    def set_animation(self, name):
        if self.skeleton:
            self.skeleton.set_animation(name)

    def update(self, dt):
        if self.skeleton:
            self.skeleton.update(dt)

    # ---------- 渲染 ----------
    def render(self, width, height, scale=1.0, offset_x=0.0, offset_y=0.0,
               background=(0, 0, 0, 0)):
        canvas = np.zeros((height, width, 4), dtype=np.float32)
        if background[3] > 0:
            canvas[..., :] = (background[0] / 255.0, background[1] / 255.0,
                              background[2] / 255.0, background[3] / 255.0)
        if self.skeleton is None:
            return _to_image(canvas)

        sk = self.skeleton
        cx = width / 2.0 + offset_x
        cy = height / 2.0 + offset_y

        for si in sk._draw_order:
            slot = sk.slots[si]
            att = sk.get_attachment(slot)
            if not att:
                continue
            bone = slot['bone']
            if bone is None:
                continue
            try:
                self._draw_attachment(canvas, sk, slot, att, bone, cx, cy, scale)
            except Exception:
                continue
        return _to_image(canvas)

    @staticmethod
    def _to_world(bone, x, y):
        return (x * bone.a + y * bone.b + bone.world_x,
                x * bone.c + y * bone.d + bone.world_y)

    def _world(self, bone, x, y, cx, cy, scale):
        wx, wy = self._to_world(bone, x, y)
        return (cx + wx * scale, cy - wy * scale)

    def _draw_attachment(self, canvas, sk, slot, att, bone, cx, cy, scale):
        atype = att.get('type')
        tint = slot['color']
        blend = slot.get('blend', BLEND_NORMAL)
        deform = sk.deform_offsets(sk.default_skin, slot['name'], att.get('name'), sk.time)

        if atype == 'region':
            region = self._pick_region(att)
            if region is None:
                return
            page = region.page
            tex = self.page_arrays.get(page)
            if tex is None:
                return
            pw, ph = self.page_images[page].size
            off = _region_offset(att, region)
            uvs = _region_uvs(att, region, pw, ph)
            pts = []
            for i in range(4):
                sx, sy = self._world(bone, off[i * 2], off[i * 2 + 1], cx, cy, scale)
                pts.append((sx, sy, uvs[i][0], uvs[i][1]))
            _raster_tri(canvas, tex, pts[0], pts[1], pts[2], tint, blend)
            _raster_tri(canvas, tex, pts[0], pts[2], pts[3], tint, blend)

        elif atype == 'mesh':
            self._draw_mesh(canvas, sk, att, bone, cx, cy, scale, tint, blend, deform)

        elif atype == 'linkedmesh':
            parent = self._find_linked_parent(sk, att)
            if parent is not None:
                self._draw_mesh(canvas, sk, parent, bone, cx, cy, scale, tint, blend, deform)

    def _draw_mesh(self, canvas, sk, att, bone, cx, cy, scale, tint, blend, deform):
        region = self._pick_region(att)
        if region is None:
            return
        tex = self.page_arrays.get(region.page)
        if tex is None:
            return
        pw, ph = self.page_images[region.page].size

        worlds = mesh_world_vertices(sk, att, bone, deform)
        uvs = att.get('uvs') or []
        # uv 换算到贴图
        if region.rotate:
            # 图集里存的是未旋转尺寸,转 90° 后占的像素矩形要把宽高对调
            u_min, u_max = region.x / pw, (region.x + region.h) / pw
            v_min, v_max = region.y / ph, (region.y + region.w) / ph
        else:
            u_min, u_max = region.x / pw, (region.x + region.w) / pw
            v_min, v_max = region.y / ph, (region.y + region.h) / ph
        span_u = u_max - u_min
        span_v = v_max - v_min

        n = min(len(worlds), len(uvs) // 2)
        screen = []
        for i in range(n):
            wx, wy = worlds[i]
            sx, sy = cx + wx * scale, cy - wy * scale
            if region.rotate:
                uu = u_min + uvs[i * 2 + 1] * span_u
                vv = v_min + uvs[i * 2] * span_v
            else:
                uu = u_min + uvs[i * 2] * span_u
                vv = v_min + uvs[i * 2 + 1] * span_v
            screen.append((sx, sy, uu, vv))

        tris = att.get('triangles') or []
        for i in range(0, len(tris) - 2, 3):
            a, b, c = int(tris[i]), int(tris[i + 1]), int(tris[i + 2])
            if a >= len(screen) or b >= len(screen) or c >= len(screen):
                continue
            _raster_tri(canvas, tex, screen[a], screen[b], screen[c], tint, blend)

    def _pick_region(self, att):
        name = att.get('path') or att.get('name')
        lst = self.regions.get(name)
        if not lst:
            return None
        idx = att.get('uvs') and att.get('index')
        if len(lst) == 1:
            return lst[0]
        # 多 index 时按顺序取(简化)
        return lst[0]

    def _find_linked_parent(self, sk, att):
        skin_name = att.get('skin')
        parent_name = att.get('parent')
        if not parent_name:
            return None
        skins = [sk.skin]
        if skin_name and skin_name in sk.skins:
            skins = [sk.skins[skin_name]]
        for skin in skins:
            for sname, atts in skin.items():
                if parent_name in atts:
                    p = atts[parent_name]
                    if p.get('type') == 'mesh':
                        return p
        # 全局搜一遍
        for skin in sk.skins.values():
            for sname, atts in skin.items():
                p = atts.get(parent_name)
                if p and p.get('type') == 'mesh':
                    return p
        return None

    # ---------- 边界 ----------
    def compute_bounds(self):
        """按 setup 姿态估算所有区域附件的包围盒(用于自动缩放)"""
        if self.skeleton is None:
            return (0.0, 0.0, 1.0, 1.0)
        sk = self.skeleton
        sk.reset_pose()
        for b in sk.bones:
            b.update()
        # 不跑约束的话 IK 驱动的骨头还停在初始位置,取景会偏
        sk._apply_constraints()
        minx = miny = 1e9
        maxx = maxy = -1e9
        found = False
        for slot in sk.slots:
            att = sk.get_attachment(slot)
            if not att or slot['bone'] is None:
                continue
            bone = slot['bone']
            if att.get('type') == 'region':
                off = _region_offset(att, self._pick_region(att))
                world = [self._to_world(bone, off[i * 2], off[i * 2 + 1]) for i in range(4)]
            elif att.get('type') == 'mesh':
                world = mesh_world_vertices(sk, att, bone)
            elif att.get('type') == 'linkedmesh':
                parent = self._find_linked_parent(sk, att)
                world = mesh_world_vertices(sk, parent, bone) if parent is not None else []
            else:
                world = []
            for wx, wy in world:
                minx = min(minx, wx); maxx = max(maxx, wx)
                miny = min(miny, wy); maxy = max(maxy, wy)
                found = True
        if not found:
            return (0.0, 0.0, float(self.skeleton.width or 100), float(self.skeleton.height or 100))
        return (minx, miny, maxx, maxy)

    def apply_setup_pose(self):
        if self.skeleton:
            self.skeleton.time = 0.0
            self.skeleton.apply_pose(0.0)

    def animation_bounds(self, name, samples=12):
        """采样指定动作的包围盒,用于按动作自动取景"""
        sk = self.skeleton
        if sk is None or name not in sk.animations:
            return self.compute_bounds()
        prev = sk.animation_name
        sk.set_animation(name)
        dur = sk.duration
        n = samples if dur > 0 else 1
        minx = miny = 1e9
        maxx = maxy = -1e9
        found = False
        for i in range(n):
            sk.apply_pose((dur * i / n) if dur > 0 else 0.0)
            for slot in sk.slots:
                att = sk.get_attachment(slot)
                bone = slot['bone']
                if not att or bone is None:
                    continue
                if att.get('type') == 'region':
                    off = _region_offset(att, self._pick_region(att))
                    world = [self._to_world(bone, off[j * 2], off[j * 2 + 1]) for j in range(4)]
                elif att.get('type') == 'mesh':
                    world = mesh_world_vertices(sk, att, bone)
                elif att.get('type') == 'linkedmesh':
                    parent = self._find_linked_parent(sk, att)
                    world = mesh_world_vertices(sk, parent, bone) if parent is not None else []
                else:
                    world = []
                for wx, wy in world:
                    minx = min(minx, wx); maxx = max(maxx, wx)
                    miny = min(miny, wy); maxy = max(maxy, wy)
                    found = True
        sk.set_animation(prev)
        if not found:
            return self.compute_bounds()
        return (minx, miny, maxx, maxy)


def _to_image(arr):
    a = np.clip(arr, 0.0, 1.0)
    rgb = (a[..., :3] * 255.0).astype(np.uint8)
    alpha = (a[..., 3] * 255.0).astype(np.uint8)
    out = np.dstack([rgb, alpha])
    return Image.fromarray(out, 'RGBA')


# ============================ 定位辅助 ============================

def _find_file(directory, filename):
    low = filename.lower()
    try:
        for fn in os.listdir(directory):
            if fn.lower() == low:
                return os.path.join(directory, fn)
    except Exception:
        pass
    return None


def _locate(path):
    """返回 (目录, 骨架文件, 图集文件)"""
    if os.path.isfile(path):
        directory = os.path.dirname(path)
        ext = os.path.splitext(path)[1].lower()
        skel = path if ext in ('.json', '.skel') else None
    else:
        directory = path
        skel = None

    atlases = []
    skel_candidates = []
    try:
        for fn in sorted(os.listdir(directory)):
            low = fn.lower()
            if low.endswith('.atlas'):
                atlases.append(os.path.join(directory, fn))
            elif low.endswith('.json') and not low.endswith('.atlas.json'):
                skel_candidates.append(os.path.join(directory, fn))
            elif low.endswith('.skel') and skel is None:
                skel_candidates.append(os.path.join(directory, fn))
    except Exception:
        pass

    atlas = None
    if skel is not None:
        # 已经指定了骨架,就找跟它同名的图集。
        # 素材目录里塞了几百套,不能随手拿第一个,不然纹理全对不上,画面一片空白
        base = os.path.splitext(os.path.basename(skel))[0].lower()
        for a in atlases:
            if os.path.splitext(os.path.basename(a))[0].lower() == base:
                atlas = a
                break
    if atlas is None and atlases:
        atlas = atlases[0]

    if skel is None and skel_candidates:
        # 没指定骨架时,优先与 atlas 同名的
        if atlas:
            base = os.path.splitext(os.path.basename(atlas))[0].lower()
            for c in skel_candidates:
                if os.path.splitext(os.path.basename(c))[0].lower() == base:
                    skel = c
                    break
        if skel is None:
            skel = skel_candidates[0]
    return directory, skel, atlas