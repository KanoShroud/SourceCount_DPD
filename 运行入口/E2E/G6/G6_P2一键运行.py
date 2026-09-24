"""PyCharm直接运行：G6-P2复用冻结缓存、六轨训练及最终比较。"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

if __name__ == '__main__':
    if Path(sys.executable).resolve() != Path('D:/Software/anaconda3/envs/PyTorch/python.exe').resolve():
        raise RuntimeError('请选择项目PyTorch解释器')
    from 统一模型代码.gates.g6.g6_p2 import main
    if len(sys.argv) == 1:
        sys.argv.append('--run')
    main()
