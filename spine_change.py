import os
import sys
import json
import math
import struct
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from spine_render import _locate

from spine_web import ViewerServer, open_viewer


# ======================== Spine 3.3-3.6 Binary Parser ========================
# 移植自 skb3.3-6.js 的 SkeletonBinary3

BLEND_MODE = ["normal", "additive", "multiply", "screen"]
ATTACHMENT_TYPE = ["region", "boundingbox", "mesh", "linkedmesh", "path", "point", "clipping"]
TRANSFORM_MODE = ["normal", "onlyTranslation", "noRotationOrReflection", "noScale", "noScaleOrReflection"]
POSITION_MODE = ["fixed", "percent"]
SPACING_MODE = ["length", "fixed", "percent"]
ROTATE_MODE = ["tangent", "chain", "chainScale"]


class SpineBinaryReader:
    """读取 Spine 3.3-3.6 的 .skel 二进制文件并解析为 JSON 字典"""

    def __init__(self):
        self.data = None
        self.scale = 1
        self.json = {}
        self.next_num = 0
        self.chars = ""
        self.ver_minor = 0  # 当前解析的小版本号,3.7 起格式有变化

    def read_byte(self):
        if self.next_num < len(self.data):
            b = self.data[self.next_num]
            self.next_num += 1
            return b
        return 0

    def read_boolean(self):
        return self.read_byte() != 0

    def read_short(self):
        return (self.read_byte() << 8) | self.read_byte()

    def read_int(self, optimize_positive=None):
        if optimize_positive is None:
            # 4字节大端有符号整数
            b0 = self.read_byte()
            b1 = self.read_byte()
            b2 = self.read_byte()
            b3 = self.read_byte()
            val = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
            # 转为有符号32位
            if val & 0x80000000:
                val -= 0x100000000
            return val

        # varint 读取
        b = self.read_byte()
        result = b & 0x7f
        if (b & 0x80) != 0:
            b = self.read_byte()
            result |= (b & 0x7f) << 7
            if (b & 0x80) != 0:
                b = self.read_byte()
                result |= (b & 0x7f) << 14
                if (b & 0x80) != 0:
                    b = self.read_byte()
                    result |= (b & 0x7f) << 21
                    if (b & 0x80) != 0:
                        b = self.read_byte()
                        result |= (b & 0x7f) << 28

        if optimize_positive:
            return result
        else:
            # zigzag 解码
            return (result >> 1) ^ (-(result & 1))

    def read_float(self):
        raw = bytes([self.read_byte(), self.read_byte(), self.read_byte(), self.read_byte()])
        return struct.unpack('>f', raw)[0]

    def read_string(self):
        char_count = self.read_int(True)
        if char_count == 0:
            return None
        if char_count == 1:
            return ""
        char_count -= 1
        raw = bytearray()
        for _ in range(char_count):
            raw.append(self.read_byte())
        return raw.decode('utf-8')

    def read_hex(self):
        b = self.read_byte()
        return format(b, '02x')

    def read_color(self):
        return self.read_hex() + self.read_hex() + self.read_hex() + self.read_hex()

    def read_color_packed(self):
        # 3.7 开始颜色是打包成 4 字节的 RGBA8888 int
        v = self.read_int(None) & 0xffffffff
        return format(v, '08x')

    def read_color_ver(self):
        # 按版本选择颜色读法,统一返回 "rrggbbaa" 字符串
        return self.read_color_packed() if self.ver_minor >= 7 else self.read_color()

    def read_float_array(self, n, scale):
        array = []
        if scale == 1:
            for _ in range(n):
                array.append(self.read_float())
        else:
            for _ in range(n):
                array.append(self.read_float() * scale)
        return array

    def read_short_array(self):
        n = self.read_int(True)
        array = []
        for _ in range(n):
            array.append(self.read_short())
        return array

    def read_vertices(self, vertex_count):
        vertices_length = vertex_count << 1
        if not self.read_boolean():
            return self.read_float_array(vertices_length, self.scale)

        vertices = []
        for _ in range(vertex_count):
            bone_count = self.read_int(True)
            vertices.append(bone_count)
            for _ in range(bone_count):
                vertices.append(self.read_int(True))
                vertices.append(self.read_float() * self.scale)
                vertices.append(self.read_float() * self.scale)
                vertices.append(self.read_float())
        return vertices

    # ---- 版本相关的条件读取 ----
    def check_ik(self, minor_ver, reader, obj):
        if minor_ver >= 5:
            obj['order'] = reader.read_int(True)

    def check_trs(self, minor_ver, reader, obj):
        if minor_ver >= 5:
            obj['order'] = reader.read_int(True)

    def check_path(self, minor_ver, reader, obj):
        if minor_ver >= 5:
            obj['order'] = reader.read_int(True)

    def check_skel(self, minor_ver, reader, skeleton, nonessential):
        if nonessential is not None and nonessential is not False:
            if minor_ver >= 5:
                skeleton['fps'] = reader.read_float()
            a = reader.read_string()
            skeleton['images'] = None if a == '' or a is None else a
            if minor_ver >= 7:
                # 3.7 头部多了音频目录
                b = reader.read_string()
                skeleton['audio'] = None if b == '' or b is None else b

    def read_bones(self, minor_ver, reader, bones, nonessential):
        scale = reader.scale
        for l in range(len(bones)):
            bone = {}
            bone['name'] = reader.read_string()
            bone['parent'] = None

            if l > 0:
                parent_index = reader.read_int(True)
                bone['parent'] = bones[parent_index]['name']

            rotation = reader.read_float()
            if rotation != 0:
                bone['rotation'] = rotation
            x = reader.read_float()
            if x != 0:
                bone['x'] = x * scale
            y = reader.read_float()
            if y != 0:
                bone['y'] = y * scale
            scale_x = reader.read_float()
            if scale_x != 1:
                bone['scaleX'] = scale_x
            scale_y = reader.read_float()
            if scale_y != 1:
                bone['scaleY'] = scale_y
            shear_x = reader.read_float()
            if shear_x != 0:
                bone['shearX'] = shear_x
            shear_y = reader.read_float()
            if shear_y != 0:
                bone['shearY'] = shear_y
            length = reader.read_float()
            if length != 0:
                bone['length'] = length * scale

            if minor_ver <= 4:
                bone['inheritRotation'] = reader.read_boolean()
                bone['inheritScale'] = reader.read_boolean()
            else:
                transform_index = reader.read_int(True)
                bone['transform'] = TRANSFORM_MODE[transform_index]

            if nonessential is not None and nonessential is not False:
                bone['color'] = reader.read_color_ver()

            bones[l] = bone

    def read_slots(self, minor_ver, reader, bones, slots):
        for slot_index in range(len(slots)):
            slot = {}
            slot['name'] = reader.read_string()
            bone_index = reader.read_int(True)
            slot['bone'] = bones[bone_index]['name']

            main_color = reader.read_color_ver()
            if main_color != "ffffffff":
                slot['color'] = main_color

            if minor_ver >= 7:
                # 3.7 暗部颜色无条件读取
                dark_color = reader.read_color_packed()
                if dark_color != "ffffffff":
                    slot['dark'] = dark_color
            elif minor_ver >= 6:
                dark_color = reader.read_color()
                if dark_color != "ffffffff":
                    slot['dark'] = dark_color

            slot['attachment'] = reader.read_string()
            blend_index = reader.read_int(True)
            slot['blend'] = BLEND_MODE[blend_index]
            slots[slot_index] = slot

    def read_skin(self, skin_name, nonessential):
        slot_count = self.read_int(True)
        if slot_count == 0:
            return None
        skin = {}
        for _ in range(slot_count):
            slot_index = self.read_int(True)
            slot = {}
            n = self.read_int(True)
            for _ in range(n):
                name = self.read_string()
                attachment = self.read_attachment(name, nonessential)
                if attachment is not None:
                    slot[name] = attachment
            skin[self.json['slots'][slot_index]['name']] = slot
        return skin

    def read_attachment(self, attachment_name, nonessential):
        scale = self.scale
        # 原JS用未定义的 this.nonessential,可选颜色一直不读,导致含非必要数据的文件错位;这里按规范用传入值
        opt = bool(nonessential)
        name = self.read_string()
        if name is None:
            name = attachment_name
        att_type = ATTACHMENT_TYPE[self.read_byte()]

        if att_type == "region":
            path = self.read_string()
            if path is None:
                path = name
            region = {}
            region['type'] = "region"
            region['name'] = name
            region['path'] = path.strip()
            region['rotation'] = self.read_float()
            region['x'] = self.read_float() * scale
            region['y'] = self.read_float() * scale
            region['scaleX'] = self.read_float()
            region['scaleY'] = self.read_float()
            region['width'] = self.read_float() * scale
            region['height'] = self.read_float() * scale
            region['color'] = self.read_color_ver()
            return region

        elif att_type == "boundingbox":
            box = {}
            box['type'] = "boundingbox"
            box['name'] = name
            vertex_count = self.read_int(True)
            box['vertexCount'] = vertex_count
            box['vertices'] = self.read_vertices(vertex_count)
            if opt:
                box['color'] = self.read_color_ver()
            return box

        elif att_type == "mesh":
            path = self.read_string()
            if path is None:
                path = name
            mesh = {}
            mesh['type'] = "mesh"
            mesh['name'] = name
            mesh['path'] = path
            mesh['color'] = self.read_color_ver()
            vertex_count = self.read_int(True)
            mesh['uvs'] = self.read_float_array(vertex_count << 1, 1)
            mesh['triangles'] = self.read_short_array()
            mesh['vertices'] = self.read_vertices(vertex_count)
            mesh['hull'] = self.read_int(True)
            if nonessential:
                mesh['edges'] = self.read_short_array()
                mesh['width'] = self.read_float() * scale
                mesh['height'] = self.read_float() * scale
            return mesh

        elif att_type == "linkedmesh":
            path = self.read_string()
            if path is None:
                path = name
            linked = {}
            linked['type'] = "linkedmesh"
            linked['name'] = name
            linked['path'] = path
            linked['color'] = self.read_color_ver()
            linked['skin'] = self.read_string()
            linked['parent'] = self.read_string()
            linked['deform'] = self.read_boolean()
            if nonessential:
                linked['width'] = self.read_float() * scale
                linked['height'] = self.read_float() * scale
            return linked

        elif att_type == "path":
            path = {}
            path['type'] = "path"
            path['name'] = name
            path['closed'] = self.read_boolean()
            path['constantSpeed'] = self.read_boolean()
            vertex_count = self.read_int(True)
            path['vertexCount'] = vertex_count
            path['vertices'] = self.read_vertices(vertex_count)
            lengths = []
            for _ in range(int(vertex_count / 3)):
                lengths.append(self.read_float() * scale)
            path['lengths'] = lengths
            if opt:
                path['color'] = self.read_color_ver()
            return path

        elif att_type == "point":
            point = {}
            point['type'] = "point"
            point['name'] = name
            point['rotation'] = self.read_float()
            point['x'] = self.read_float() * scale
            point['y'] = self.read_float() * scale
            if opt:
                point['color'] = self.read_color_ver()
            return point

        elif att_type == "clipping":
            clipping = {}
            clipping['type'] = "clipping"
            clipping['name'] = name
            clipping['end'] = self.read_int(True)
            vertex_count = self.read_int(True)
            clipping['vertexCount'] = vertex_count
            clipping['vertices'] = self.read_vertices(vertex_count)
            if opt:
                clipping['color'] = self.read_color_ver()
            return clipping

        return None

    def read_curve(self, frame_index, timeline):
        curve_type = self.read_byte()
        if curve_type == 0:
            # linear, 什么都不做
            pass
        elif curve_type == 1:
            timeline[frame_index]['curve'] = "stepped"
        elif curve_type == 2:
            cx1 = self.read_float()
            cy1 = self.read_float()
            cx2 = self.read_float()
            cy2 = self.read_float()
            timeline[frame_index]['curve'] = [cx1, cy1, cx2, cy2]

    def read_animation(self, name):
        animation = {}
        scale = self.scale
        duration = 0

        # ---- Slot timelines ----
        slots = {}
        n = self.read_int(True)
        for _ in range(n):
            slot_index = self.read_int(True)
            slot_map = {}
            time_count = self.read_int(True)
            for _ in range(time_count):
                timeline_type = self.read_byte()
                frame_count = self.read_int(True)

                if timeline_type == 0:  # SLOT_ATTACHMENT
                    timeline = []
                    for fi in range(frame_count):
                        t = self.read_float()
                        att_name = self.read_string()
                        frame = {'time': t, 'name': att_name}
                        timeline.append(frame)
                    slot_map['attachment'] = timeline
                    duration = max(duration, timeline[frame_count - 1]['time'])

                elif timeline_type == 1:  # SLOT_COLOR
                    timeline = []
                    for fi in range(frame_count):
                        t = self.read_float()
                        color = self.read_color_ver()
                        frame = {'time': t, 'color': color}
                        timeline.append(frame)
                        if fi < frame_count - 1:
                            self.read_curve(fi, timeline)
                    slot_map['color'] = timeline
                    duration = max(duration, timeline[frame_count - 1]['time'])

                elif timeline_type == 2:  # SLOT_TWO_COLOR
                    timeline = []
                    for fi in range(frame_count):
                        t = self.read_float()
                        light = self.read_color_ver()
                        dark = self.read_color_ver()
                        frame = {'time': t, 'light': light, 'dark': dark}
                        timeline.append(frame)
                        if fi < frame_count - 1:
                            self.read_curve(fi, timeline)
                    slot_map['twoColor'] = timeline
                    duration = max(duration, timeline[frame_count - 1]['time'])

            slots[self.json['slots'][slot_index]['name']] = slot_map
        animation['slots'] = slots

        # ---- Bone timelines ----
        bones = {}
        n = self.read_int(True)
        for _ in range(n):
            bone_index = self.read_int(True)
            bone_map = {}
            nn = self.read_int(True)
            for _ in range(nn):
                timeline_type = self.read_byte()
                frame_count = self.read_int(True)

                if timeline_type == 0:  # BONE_ROTATE
                    timeline = []
                    for fi in range(frame_count):
                        t = self.read_float()
                        angle = self.read_float()
                        frame = {'time': t, 'angle': angle}
                        timeline.append(frame)
                        if fi < frame_count - 1:
                            self.read_curve(fi, timeline)
                    bone_map['rotate'] = timeline
                    duration = max(duration, timeline[frame_count - 1]['time'])

                elif timeline_type in (1, 2, 3):  # BONE_TRANSLATE / SCALE / SHEAR
                    timeline = []
                    timeline_scale = scale if timeline_type == 1 else 1
                    for fi in range(frame_count):
                        t = self.read_float()
                        x = self.read_float()
                        y = self.read_float()
                        frame = {'time': t, 'x': x * timeline_scale, 'y': y * timeline_scale}
                        timeline.append(frame)
                        if fi < frame_count - 1:
                            self.read_curve(fi, timeline)
                    if timeline_type == 1:
                        bone_map['translate'] = timeline
                    elif timeline_type == 2:
                        bone_map['scale'] = timeline
                    else:
                        bone_map['shear'] = timeline
                    duration = max(duration, timeline[frame_count - 1]['time'])

            bones[self.json['bones'][bone_index]['name']] = bone_map
        animation['bones'] = bones

        # ---- IK timelines ----
        ik = {}
        n = self.read_int(True)
        for _ in range(n):
            ik_index = self.read_int(True)
            frame_count = self.read_int(True)
            timeline = []
            for fi in range(frame_count):
                t = self.read_float()
                mix = self.read_float()
                bend_positive = self.read_byte() != 255
                frame = {'time': t, 'mix': mix, 'bendPositive': bend_positive}
                if self.ver_minor >= 7:
                    # 3.7 每帧多带压缩/拉伸
                    frame['compress'] = self.read_boolean()
                    frame['stretch'] = self.read_boolean()
                timeline.append(frame)
                if fi < frame_count - 1:
                    self.read_curve(fi, timeline)
            ik[self.json['ik'][ik_index]['name']] = timeline
            duration = max(duration, timeline[frame_count - 1]['time'])
        animation['ik'] = ik

        # ---- Transform timelines ----
        transform = {}
        n = self.read_int(True)
        for _ in range(n):
            transform_index = self.read_int(True)
            frame_count = self.read_int(True)
            timeline = []
            for fi in range(frame_count):
                frame = {
                    'time': self.read_float(),
                    'rotateMix': self.read_float(),
                    'translateMix': self.read_float(),
                    'scaleMix': self.read_float(),
                    'shearMix': self.read_float()
                }
                timeline.append(frame)
                if fi < frame_count - 1:
                    self.read_curve(fi, timeline)
            transform[self.json['transform'][transform_index]['name']] = timeline
            duration = max(duration, timeline[frame_count - 1]['time'])
        animation['transform'] = transform

        # ---- Path timelines ----
        paths = {}
        n = self.read_int(True)
        for _ in range(n):
            path_index = self.read_int(True)
            path_const = self.json['path'][path_index]
            path_map = {}
            nn = self.read_int(True)
            for _ in range(nn):
                timeline_type = self.read_byte()
                frame_count = self.read_int(True)

                if timeline_type in (0, 1):  # PATH_POSITION / PATH_SPACING
                    timeline = []
                    timeline_scale = 1
                    if timeline_type == 1:
                        if path_const.get('spacingMode') in ("length", "fixed"):
                            timeline_scale = self.scale
                    else:
                        if path_const.get('positionMode') == "fixed":
                            timeline_scale = self.scale
                    for fi in range(frame_count):
                        t = self.read_float()
                        f = self.read_float()
                        frame = {'time': t}
                        if timeline_type == 0:
                            frame['position'] = f * timeline_scale
                        else:
                            frame['spacing'] = f * timeline_scale
                        timeline.append(frame)
                        if fi < frame_count - 1:
                            self.read_curve(fi, timeline)
                    if timeline_type == 0:
                        path_map['position'] = timeline
                    else:
                        path_map['spacing'] = timeline
                    duration = max(duration, timeline[frame_count - 1]['time'])

                elif timeline_type == 2:  # PATH_MIX
                    timeline = []
                    for fi in range(frame_count):
                        t = self.read_float()
                        rotate_mix = self.read_float()
                        translate_mix = self.read_float()
                        frame = {'time': t, 'rotateMix': rotate_mix, 'translateMix': translate_mix}
                        timeline.append(frame)
                        if fi < frame_count - 1:
                            self.read_curve(fi, timeline)
                    path_map['mix'] = timeline
                    duration = max(duration, timeline[frame_count - 1]['time'])

            paths[self.json['path'][path_index]['name']] = path_map
        animation['paths'] = paths

        # ---- Deform timelines ----
        deform = {}
        n = self.read_int(True)
        for _ in range(n):
            skin_index = self.read_int(True)
            skin_name = self.json['skinsName'][skin_index]
            skin = {}
            nn = self.read_int(True)
            for _ in range(nn):
                slot_index = self.read_int(True)
                slot_att = self.json['slots'][slot_index]
                slot = {}
                nnn = self.read_int(True)
                for _ in range(nnn):
                    mesh_name = self.read_string()
                    frame_count = self.read_int(True)
                    timeline = []
                    for fi in range(frame_count):
                        t = self.read_float()
                        end = self.read_int(True)
                        frame = {'time': t}
                        if end != 0:
                            vertices = []
                            start = self.read_int(True)
                            if self.scale == 1:
                                for _ in range(end):
                                    vertices.append(self.read_float())
                            else:
                                for _ in range(end):
                                    vertices.append(self.read_float() * self.scale)
                            frame['offset'] = start
                            frame['vertices'] = vertices
                        timeline.append(frame)
                        if fi < frame_count - 1:
                            self.read_curve(fi, timeline)
                    slot[mesh_name] = timeline
                    duration = max(duration, timeline[frame_count - 1]['time'])
                skin[slot_att['name']] = slot
            deform[skin_name] = skin
        animation['deform'] = deform

        # ---- Draw order timeline ----
        draw_order_count = self.read_int(True)
        if draw_order_count > 0:
            draw_orders = []
            for _ in range(draw_order_count):
                draw_order_map = {}
                time = self.read_float()
                offset_count = self.read_int(True)
                offsets = []
                for _ in range(offset_count):
                    offset_map = {}
                    slot_index = self.read_int(True)
                    offset_map['slot'] = self.json['slots'][slot_index]['name']
                    dooffset = self.read_int(True)
                    offset_map['offset'] = dooffset
                    offsets.append(offset_map)
                draw_order_map['offsets'] = offsets
                draw_order_map['time'] = time
                draw_orders.append(draw_order_map)
            duration = max(duration, draw_orders[draw_order_count - 1]['time'])
            animation['drawOrder'] = draw_orders

        # ---- Event timeline ----
        event_count = self.read_int(True)
        if event_count > 0:
            events = []
            for _ in range(event_count):
                time = self.read_float()
                event_name = self.json['eventsName'][self.read_int(True)]
                event_data = self.json['events'][event_name]
                e = {
                    'name': event_name,
                    'int': self.read_int(False),
                    'float': self.read_float(),
                    'string': self.read_string() if self.read_boolean() else event_data['string'],
                    'time': time
                }
                if self.ver_minor >= 7 and event_data.get('audioPath'):
                    # 带音频的事件多读音量/左右声道
                    e['volume'] = self.read_float()
                    e['balance'] = self.read_float()
                events.append(e)
            duration = max(duration, events[event_count - 1]['time'])
            animation['events'] = events

        return animation

    def init_json(self):
        """主解析入口,读取整个 .skel 二进制并构建 JSON 字典"""
        self.json['skeleton'] = {}
        skeleton = self.json['skeleton']
        skeleton['hash'] = self.read_string()
        if skeleton['hash'] is not None and len(skeleton['hash']) == 0:
            skeleton['hash'] = None
        skeleton['spine'] = self.read_string()
        if skeleton['spine'] is not None and len(skeleton['spine']) == 0:
            skeleton['spine'] = None
        skeleton['width'] = self.read_float()
        skeleton['height'] = self.read_float()
        nonessential = self.read_boolean()

        version = skeleton['spine'] or "0.0.0"
        parts = version.split('.')
        major_version = int(parts[0]) if parts[0].isdigit() else 0
        minor_version = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

        if major_version == 3 and minor_version == 8:
            raise ValueError(f"此解析器不支持 Spine {version},仅支持 3.3-3.7 的 .skel 文件")

        self.ver_minor = minor_version
        self.check_skel(minor_version, self, skeleton, nonessential)

        # Bones
        self.json['bones'] = [None] * self.read_int(True)
        bones = self.json['bones']
        self.read_bones(minor_version, self, bones, nonessential)

        # Slots
        self.json['slots'] = [None] * self.read_int(True)
        slots = self.json['slots']
        self.read_slots(minor_version, self, bones, slots)

        # IK constraints
        self.json['ik'] = [None] * self.read_int(True)
        ik = self.json['ik']
        for i in range(len(ik)):
            ik_c = {}
            ik_c['name'] = self.read_string()
            self.check_ik(minor_version, self, ik_c)
            ik_c['bones'] = [None] * self.read_int(True)
            for j in range(len(ik_c['bones'])):
                ik_c['bones'][j] = bones[self.read_int(True)]['name']
            ik_c['target'] = bones[self.read_int(True)]['name']
            ik_c['mix'] = self.read_float()
            ik_c['bendPositive'] = self.read_byte() != 255
            if minor_version >= 7:
                # 3.7 的 IK 约束多了压缩/拉伸/统一缩放
                ik_c['compress'] = self.read_boolean()
                ik_c['stretch'] = self.read_boolean()
                ik_c['uniform'] = self.read_boolean()
            ik[i] = ik_c

        # Transform constraints
        self.json['transform'] = [None] * self.read_int(True)
        transform = self.json['transform']
        for i in range(len(transform)):
            tc = {}
            tc['name'] = self.read_string()
            self.check_trs(minor_version, self, tc)
            tc_bones = [None] * self.read_int(True)
            for ii in range(len(tc_bones)):
                tc_bones[ii] = bones[self.read_int(True)]['name']
            tc['bones'] = tc_bones
            tc['target'] = bones[self.read_int(True)]['name']
            if minor_version >= 6:
                tc['local'] = self.read_boolean()
                tc['relative'] = self.read_boolean()
            tc['rotation'] = self.read_float()
            tc['x'] = self.read_float() * self.scale
            tc['y'] = self.read_float() * self.scale
            tc['scaleX'] = self.read_float()
            tc['scaleY'] = self.read_float()
            tc['shearY'] = self.read_float()
            tc['rotateMix'] = self.read_float()
            tc['translateMix'] = self.read_float()
            tc['scaleMix'] = self.read_float()
            tc['shearMix'] = self.read_float()
            transform[i] = tc

        # Path constraints
        self.json['path'] = [None] * self.read_int(True)
        path = self.json['path']
        for i in range(len(path)):
            pc = {}
            pc['name'] = self.read_string()
            self.check_path(minor_version, self, pc)
            pc['bones'] = [None] * self.read_int(True)
            for ii in range(len(pc['bones'])):
                pc['bones'][ii] = bones[self.read_int(True)]['name']
            pc['target'] = slots[self.read_int(True)]['name']
            pc['positionMode'] = POSITION_MODE[self.read_int(True)]
            pc['spacingMode'] = SPACING_MODE[self.read_int(True)]
            pc['rotateMode'] = ROTATE_MODE[self.read_int(True)]
            pc['rotation'] = self.read_float()
            pc['position'] = self.read_float()
            if pc['positionMode'] == "fixed":
                pc['position'] *= self.scale
            pc['spacing'] = self.read_float()
            if pc['spacingMode'] in ("length", "fixed"):
                pc['spacing'] *= self.scale
            pc['rotateMix'] = self.read_float()
            pc['translateMix'] = self.read_float()
            path[i] = pc

        # Skins
        self.json['skins'] = {}
        self.json['skinsName'] = []
        skins = self.json['skins']
        default_skin = self.read_skin("default", nonessential)
        if default_skin is not None:
            skins['default'] = default_skin
            self.json['skinsName'].append("default")

        n = self.read_int(True)
        for _ in range(n):
            skin_name = self.read_string()
            skin = self.read_skin(skin_name, nonessential)
            skins[skin_name] = skin
            self.json['skinsName'].append(skin_name)

        # Events
        self.json['events'] = {}
        self.json['eventsName'] = []
        events = self.json['events']
        n = self.read_int(True)
        for i in range(n):
            event_name = self.read_string()
            event = {}
            event['int'] = self.read_int(False)
            event['float'] = self.read_float()
            event['string'] = self.read_string()
            if minor_version >= 7:
                # 3.7 事件多了音频路径,有路径时再读音量/左右声道
                audio_path = self.read_string()
                if audio_path:
                    event['audioPath'] = audio_path
                    event['volume'] = self.read_float()
                    event['balance'] = self.read_float()
            events[event_name] = event
            self.json['eventsName'].append(event_name)

        # Animations
        self.json['animations'] = {}
        animations = self.json['animations']
        n = self.read_int(True)
        for _ in range(n):
            anim_name = self.read_string()
            animations[anim_name] = self.read_animation(anim_name)

    def parse(self, data, scale=1):
        """从字节数据解析"""
        self.data = data
        self.scale = scale
        self.next_num = 0
        self.init_json()
        return self.json


def detect_skel_version(data):
    """只读取 .skel 文件头部,获取 spine 版本号"""
    reader = SpineBinaryReader()
    reader.data = data
    reader.scale = 1
    # 读 hash
    hash_str = reader.read_string()
    # 读 version
    version = reader.read_string()
    return version


# ======================== Spine 3.8 转换器 ========================
# 移植自 skb3.3-6.js 的 cv4spine38 及其辅助函数

class Spine38Converter:
    """将任意版本的 Spine JSON 数据转换为 3.8 格式"""

    @staticmethod
    def remove_parent_null(obj):
        """递归删除 parent: null 的属性"""
        if obj is None:
            return None
        if isinstance(obj, list):
            for item in obj:
                Spine38Converter.remove_parent_null(item)
            return None
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                value = obj[key]
                if key == "parent" and value is None:
                    del obj[key]
                else:
                    Spine38Converter.remove_parent_null(value)
            return None
        return obj

    @staticmethod
    def fix_skins(skins):
        """将 skinnedmesh 类型改为 mesh"""
        if not isinstance(skins, dict):
            return
        for skin_name in skins:
            skin = skins[skin_name]
            if not isinstance(skin, dict):
                continue
            for att_name in skin:
                attachment = skin[att_name]
                if isinstance(attachment, dict) and attachment.get('type') == "skinnedmesh":
                    attachment['type'] = "mesh"

    @staticmethod
    def fix_skins_st(data):
        """将 skins 对象转为数组格式,删除 skinsName"""
        skins = data.get('skins')
        if not isinstance(skins, dict):
            return
        data['skins'] = [
            {'name': name, 'attachments': attachments}
            for name, attachments in skins.items()
        ]
        if 'skinsName' in data:
            del data['skinsName']

    @staticmethod
    def fix_curve(obj):
        """修复曲线数据:数组拆成 curve/c2/c3/c4,删除 "linear" curve"""
        if obj is None:
            return None
        if isinstance(obj, list):
            for item in obj:
                Spine38Converter.fix_curve(item)
            return None
        if isinstance(obj, dict):
            if 'curve' in obj:
                curve = obj['curve']
                if isinstance(curve, list):
                    obj['curve'] = curve[0] if len(curve) > 0 else None
                    if len(curve) > 1:
                        obj['c2'] = curve[1]
                    if len(curve) > 2:
                        obj['c3'] = curve[2]
                    if len(curve) > 3:
                        obj['c4'] = curve[3]
                elif curve == "linear":
                    del obj['curve']
            for value in list(obj.values()):
                Spine38Converter.fix_curve(value)
            return None
        return None

    @staticmethod
    def format_spine_decimals(data):
        """递归处理所有数字,保留3位小数"""
        if isinstance(data, list):
            return [Spine38Converter.format_spine_decimals(item) for item in data]
        if isinstance(data, dict):
            return {k: Spine38Converter.format_spine_decimals(v) for k, v in data.items()}
        if isinstance(data, (int, float)):
            import math
            if math.isnan(data) or math.isinf(data):
                return data
            rounded = round(data * 1000) / 1000
            # 去掉浮点精度问题 (如 0.30000000000000004)
            return round(rounded, 3)
        return data

    @staticmethod
    def process_json_end_and_name(data):
        """递归处理:对有 end(number) 和 name 的对象,把 name 值赋给 end,删除 name"""
        if isinstance(data, list):
            return [Spine38Converter.process_json_end_and_name(item) for item in data]
        if isinstance(data, dict):
            new_obj = dict(data)
            has_valid_end = 'end' in new_obj and isinstance(new_obj['end'], (int, float))
            has_name = 'name' in new_obj
            if has_valid_end and has_name:
                new_obj['end'] = new_obj['name']
                del new_obj['name']
            for key in list(new_obj.keys()):
                new_obj[key] = Spine38Converter.process_json_end_and_name(new_obj[key])
            return new_obj
        return data

    @staticmethod
    def convert(json_data):
        """主转换函数,将任意版本的 Spine JSON 转为 3.8 格式"""
        a = json_data

        # 判断版本号小数部分是否 >= 6 (复刻原JS的3进制parseInt逻辑)
        spine_ver = a.get('skeleton', {}).get('spine', None) if isinstance(a.get('skeleton'), dict) else None
        if spine_ver:
            parts = spine_ver.split('.')
            minor_str = parts[1] if len(parts) > 1 else None
        else:
            minor_str = None

        # 原JS: parseInt(c, 3) 对 >=3 的数字返回 NaN,导致 c=true
        # 实际效果: 3.0/3.1/3.2 → false, 3.3+ → true
        if minor_str is None:
            c = True
        else:
            try:
                parsed = int(minor_str, 3)  # base 3
                c = not (6 > parsed)
            except ValueError:
                c = True

        Spine38Converter.remove_parent_null(a)
        if c:
            Spine38Converter.fix_skins(a.get('skins'))
        Spine38Converter.fix_skins_st(a)
        Spine38Converter.fix_curve(a.get('animations'))
        a = Spine38Converter.format_spine_decimals(a)
        a = Spine38Converter.process_json_end_and_name(a)

        skeleton = a.get('skeleton')
        if skeleton:
            skeleton['spine'] = "3.8"
        else:
            a['skeleton'] = {'spine': "3.8"}

        return a


# ======================== 文件转换逻辑 ========================

def convert_file(input_path, output_path):
    """
    转换单个文件。返回 (success: bool, message: str)
    """
    ext = os.path.splitext(input_path)[1].lower()

    try:
        if ext == '.skel':
            with open(input_path, 'rb') as f:
                data = f.read()

            # 先检测版本
            version = detect_skel_version(data)
            if version:
                parts = version.split('.')
                major = int(parts[0]) if parts[0].isdigit() else 0
                minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

                if major == 3 and minor == 8:
                    # 已经是3.8,直接拷贝
                    shutil.copy2(input_path, output_path)
                    return True, f"已是3.8版本,直接拷贝: {os.path.basename(input_path)}"
                elif major == 3 and minor in (3, 4, 5, 6, 7):
                    # 解析二进制为JSON
                    reader = SpineBinaryReader()
                    reader.parse(data, scale=1)
                    json_data = reader.json
                    # 转换为3.8
                    json_data = Spine38Converter.convert(json_data)
                    # 写出JSON (替换扩展名为.json)
                    out_json = os.path.splitext(output_path)[0] + '.json'
                    with open(out_json, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, separators=(',', ':'))
                    return True, f"转换成功: {os.path.basename(input_path)} → {os.path.basename(out_json)}"
                elif major >= 4:
                    return False, f"不支持Spine {version}的.skel文件: {os.path.basename(input_path)}"
                else:
                    return False, f"未知版本 {version}: {os.path.basename(input_path)}"
            else:
                return False, f"无法读取版本号: {os.path.basename(input_path)}"

        elif ext == '.json':
            # 读取JSON
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 尝试解析为JSON
            try:
                json_data = json.loads(content)
            except json.JSONDecodeError:
                # 可能是.skel文件改了扩展名,尝试当二进制读
                with open(input_path, 'rb') as bf:
                    raw = bf.read()
                version = detect_skel_version(raw)
                if version:
                    parts = version.split('.')
                    major = int(parts[0]) if parts[0].isdigit() else 0
                    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    if major == 3 and minor in (3, 4, 5, 6, 7):
                        reader = SpineBinaryReader()
                        reader.parse(raw, scale=1)
                        json_data = reader.json
                        json_data = Spine38Converter.convert(json_data)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, ensure_ascii=False, separators=(',', ':'))
                        return True, f"转换成功(二进制伪装JSON): {os.path.basename(input_path)}"
                    elif major == 3 and minor == 8:
                        shutil.copy2(input_path, output_path)
                        return True, f"已是3.8版本,直接拷贝: {os.path.basename(input_path)}"
                    else:
                        return False, f"不支持的版本 {version}: {os.path.basename(input_path)}"
                else:
                    return False, f"JSON解析失败且非有效skel: {os.path.basename(input_path)}"

            # 正常JSON文件
            skeleton = json_data.get('skeleton', {})
            spine_ver = skeleton.get('spine', '') if isinstance(skeleton, dict) else ''

            if _is_v38(spine_ver):
                # 已经是3.8,直接拷贝
                shutil.copy2(input_path, output_path)
                return True, f"已是3.8版本,直接拷贝: {os.path.basename(input_path)}"

            if spine_ver:
                parts = spine_ver.split('.')
                major = int(parts[0]) if parts[0].isdigit() else 0
                if major >= 4:
                    return False, f"不支持Spine {spine_ver}的JSON: {os.path.basename(input_path)}"

            # 转换为3.8
            json_data = Spine38Converter.convert(json_data)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, separators=(',', ':'))
            return True, f"转换成功: {os.path.basename(input_path)}"

        else:
            # .atlas, .png, .jpg 等资源文件,直接拷贝
            shutil.copy2(input_path, output_path)
            return True, f"资源文件拷贝: {os.path.basename(input_path)}"

    except ValueError as e:
        return False, f"版本不支持: {os.path.basename(input_path)} - {e}"
    except Exception as e:
        return False, f"转换失败: {os.path.basename(input_path)} - {e}"


def convert_directory(input_dir, output_dir, log_callback, progress_callback=None):
    """
    遍历输入目录,转换所有文件到输出目录,保持目录结构
    """
    success_count = 0
    fail_count = 0
    failures = []

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            success, message = convert_file(input_path, output_path)
            if success:
                success_count += 1
            else:
                fail_count += 1
                failures.append(message)

            log_callback(message)

            if progress_callback:
                progress_callback(success_count + fail_count)

    summary = f"\n{'='*50}\n转换完成: 成功 {success_count} 个, 失败 {fail_count} 个"
    if failures:
        summary += "\n失败列表:\n" + "\n".join(failures)
    log_callback(summary)


# ======================== 预览用骨架加载 ========================

def _is_v38(version):
    """是不是 3.8.x(3.8 / 3.8.75 都算)"""
    parts = (version or '').split('.')
    return len(parts) >= 2 and parts[0] == '3' and parts[1] == '8'


def peek_spine_version(path):
    """只读原始素材的版本号(不做转换),预览里要显示转换前的版本"""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        if raw[:1] == b'{':
            data = json.loads(raw.decode('utf-8'))
            return (data.get('skeleton') or {}).get('spine', '') or ''
        return detect_skel_version(raw) or ''
    except Exception:
        return ''


def load_skeleton_json(path):
    """把 .skel / .json 骨架读成 3.8 兼容的字典,供预览渲染使用"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.skel':
        with open(path, 'rb') as f:
            raw = f.read()
        reader = SpineBinaryReader()
        reader.parse(raw, 1)
        return Spine38Converter.convert(reader.json)

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # 扩展名是 json 但其实存的是二进制
        with open(path, 'rb') as f:
            raw = f.read()
        reader = SpineBinaryReader()
        reader.parse(raw, 1)
        return Spine38Converter.convert(reader.json)

    spine_ver = (data.get('skeleton') or {}).get('spine', '')
    if _is_v38(spine_ver):
        # 3.8.x(含 3.8.75 这类小版本)已经是目标格式,直接用,别再走转换器
        return data
    return Spine38Converter.convert(data)


# ======================== Tkinter UI ========================

class PreviewPanel(ttk.Frame):
    """Spine 预览 / 转视频:渲染交给内嵌的网页 pixi-spine(WebGL),这里只负责选素材"""

    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self.skel_path = None
        self.src_ver = ''
        self.req_name = ''
        self.browser = None
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X)
        ttk.Button(top, text="打开 Spine 素材...", command=self.open_asset).pack(side=tk.LEFT)
        ttk.Button(top, text="重新加载预览", command=self.open_window).pack(side=tk.LEFT, padx=(6, 0))
        self.path_var = tk.StringVar(value="未加载素材")
        ttk.Label(top, textvariable=self.path_var, foreground="#666").pack(side=tk.LEFT, padx=8)
        self.status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.status_var, foreground="#666").pack(side=tk.LEFT, padx=8)

        # 预览区:网页窗口会被挂到这个 frame 里,看起来就跟界面是一体的
        self.stage = tk.Frame(self, bg='#1b1d24')
        self.stage.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.stage.bind('<Configure>', self._on_stage_resize)

    # ---------- 打开素材 ----------
    def open_asset(self):
        path = filedialog.askopenfilename(
            title="选择 Spine 素材",
            filetypes=[("Spine 骨架", "*.json *.skel"), ("所有文件", "*.*")])
        if path:
            self.load_asset(path)

    def load_asset(self, path):
        directory, skel_path, atlas_path = _locate(path)
        if skel_path is None or atlas_path is None:
            messagebox.showerror("加载失败", "没有找到配套的骨架文件或 .atlas 图集文件")
            return
        self.skel_path = skel_path
        self.path_var.set(skel_path)
        self.src_ver = peek_spine_version(skel_path)
        # 3.8 的二进制网页能直接读,老版本得让服务端先转成 3.8 的 json
        if os.path.splitext(skel_path)[1].lower() == '.skel' and _is_v38(self.src_ver):
            self.req_name = os.path.basename(skel_path)
        else:
            self.req_name = os.path.splitext(os.path.basename(skel_path))[0] + '.json'
        self.on_loaded()

    def on_loaded(self):
        """素材换好之后的收尾,子类可覆写"""
        self.open_window()

    def open_window(self):
        if not self.skel_path:
            messagebox.showwarning("提示", "请先打开 Spine 素材")
            return
        self.update_idletasks()
        self._close_browser()
        try:
            self.browser = open_viewer(self.app.viewer_server, self.skel_path,
                                       parent_hwnd=self.stage.winfo_id(),
                                       src_version=self.src_ver,
                                       request_name=self.req_name)
            self.status_var.set("已在内嵌预览中加载")
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    def _close_browser(self):
        if self.browser is not None:
            try:
                self.browser.close()
            except Exception:
                pass
            self.browser = None

    def _on_stage_resize(self, _event=None):
        if self.browser is not None:
            try:
                self.browser.resize()
            except Exception:
                pass


class ConvertPanel(ttk.Frame):
    """Spine 版本转换面板"""

    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        self.converting = False

        # 输入目录
        input_frame = ttk.Frame(self)
        input_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(input_frame, text="输入目录:").pack(side=tk.LEFT)
        self.input_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_frame, text="浏览...", command=self.browse_input).pack(side=tk.RIGHT)

        # 输出目录
        output_frame = ttk.Frame(self)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(output_frame, text="输出目录:").pack(side=tk.LEFT)
        self.output_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(output_frame, text="浏览...", command=self.browse_output).pack(side=tk.RIGHT)

        # 转换按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        self.convert_btn = ttk.Button(btn_frame, text="开始转换", command=self.start_convert)
        self.convert_btn.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(btn_frame, mode='indeterminate')

        # 信息栏
        info_frame = ttk.LabelFrame(self, text="转换信息")
        info_frame.pack(fill=tk.BOTH, expand=True)
        self.info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=(5, 0))

    def browse_input(self):
        d = filedialog.askdirectory(title="选择输入目录")
        if d:
            self.input_var.set(d)

    def browse_output(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.output_var.set(d)

    def log(self, message):
        self.info_text.insert(tk.END, message + "\n")
        self.info_text.see(tk.END)
        self.update_idletasks()

    def start_convert(self):
        if self.converting:
            return
        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            self.log("错误: 请选择有效的输入目录")
            return
        if not output_dir:
            self.log("错误: 请选择输出目录")
            return

        self.info_text.delete('1.0', tk.END)
        self.convert_btn.config(state=tk.DISABLED)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.progress.start()
        self.status_var.set("转换中...")
        self.converting = True

        def task():
            try:
                convert_directory(input_dir, output_dir, self.log)
            except Exception as e:
                self.log(f"致命错误: {e}")
            finally:
                self.after(0, self._convert_done)

        threading.Thread(target=task, daemon=True).start()

    def _convert_done(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.convert_btn.config(state=tk.NORMAL)
        self.status_var.set("转换完成")
        self.converting = False


class SpineChangeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spine 工具")
        self.root.geometry("1000x740")
        self.root.minsize(860, 620)

        # 预览/导出都靠这个本地服务把网页喂给浏览器,全进程共用一个
        # loader 负责把任意版本的骨架转成 3.8 再给网页,这样老版本也能预览
        self.viewer_server = ViewerServer(ask_path=self.ask_save_path,
                                          loader=load_skeleton_json)
        self._save_req = None
        self.root.after(120, self._poll_save_req)

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.preview_tab = PreviewPanel(notebook, self)
        self.convert_tab = ConvertPanel(notebook, self)

        notebook.add(self.preview_tab, text="预览 / 转视频")
        notebook.add(self.convert_tab, text="Spine转换")
        self.notebook = notebook

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def ask_save_path(self, anim):
        """网页里点导出时,http 线程把活丢过来,由主线程弹保存框;这里等它出结果"""
        box = {'path': ''}
        done = threading.Event()
        self._save_req = (anim, box, done)
        done.wait(600)
        return box['path']

    def _poll_save_req(self):
        req = self._save_req
        if req is not None:
            self._save_req = None
            anim, box, done = req
            box['path'] = filedialog.asksaveasfilename(
                title="保存 MP4", defaultextension=".mp4",
                initialfile=f"{self.viewer_server.default_name}_{anim}.mp4",
                filetypes=[("MP4 视频", "*.mp4")]) or ''
            done.set()
        self.root.after(120, self._poll_save_req)

    def on_close(self):
        self.preview_tab._close_browser()
        self.viewer_server.shutdown()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SpineChangeApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
