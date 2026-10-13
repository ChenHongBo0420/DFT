# dftpy/utils.py

import warnings
from .data_io import read_poscar

def silence_deprecation_warnings():
    """
    屏蔽掉 DeprecationWarning，避免脚本中断或刷屏。
    直接调用一次即可。
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def save_stdout_to_file(filename: str):
    """
    将 sys.stdout 重定向到指定文件。使用时：
        orig = sys.stdout
        f = open(filename, "w")
        sys.stdout = f
        # …print(…) 写入到指定文件…
        sys.stdout = orig
        f.close()
    这里我们不直接调用，而是外部根据需要手动控制。
    """
    import sys
    orig = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    return orig, f
