"""G8用户入口：点击运行正式比较/恢复；--mode short用于工程短测。不训练。"""
import os
from pathlib import Path
import sys

os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from 统一模型代码.gates.g8.runner import main  # noqa: E402

if __name__ == '__main__':
    main()
