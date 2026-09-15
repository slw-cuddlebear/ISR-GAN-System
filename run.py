# run.py
"""一键启动后端服务（Flask），同时托管前端页面，并自动打开浏览器"""
import os
import sys
import threading
import webbrowser

# 把项目根目录加入 sys.path，保证无论从哪个目录运行都能找到包
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app import app

URL = "http://127.0.0.1:5000"


def open_browser():
    webbrowser.open(URL)


if __name__ == "__main__":
    print("=" * 56)
    print("  ISR-GAN 图像超分辨率重建系统")
    print(f"  浏览器打开: {URL}")
    print("=" * 56)

    # 延迟 1.2 秒再打开浏览器，避免 Flask 还没起来
    threading.Timer(1.2, open_browser).start()

    app.run(host="0.0.0.0", port=5000, debug=False)