# SpineChange

一个 Windows 桌面小工具（Python + tkinter），用来处理 **Spine 3.3 ~ 3.6** 的动画资源：

- **格式转换**：把 3.3 ~ 3.6 的二进制 `.skel` 解析并转换成 3.8 可用的 JSON，支持单文件与整目录批量转换。
- **预览 / 转视频**：用网页 pixi-spine（WebGL）实时预览动作，支持缩放、拖拽、切换动作，并可逐帧导出为 MP4。

## 运行环境

- Python 3
- 依赖：`numpy`、`Pillow`、`imageio`、`imageio-ffmpeg`

```bash
pip install numpy Pillow imageio imageio-ffmpeg
python spine_change.py
```

预览/导出需要本机已安装 Edge 或 Chrome（浏览器 `--app` 模式打开窗口，服务只监听 `127.0.0.1`）。

## 打包成 exe

```bash
pyinstaller SpineChange.spec
```

产物在 `dist/SpineChange.exe`（已随附网页资源 `web/`）。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `spine_change.py` | 主程序：tkinter 界面、Spine 3.3-3.6 二进制解析、3.8 转换 |
| `spine_render.py` | 素材定位等渲染辅助 |
| `spine_web.py` | 本地 HTTP 小服务 + 启动浏览器预览窗口，负责收帧并合成 MP4 |
| `web/` | 预览页面与 pixi / pixi-spine 运行库 |
| `SpineChange.spec` | PyInstaller 打包配置 |

## 开源协议

本项目使用 [MIT License](LICENSE)。
