# dftpy/utils.py

import warnings
from pymatgen.io.vasp.outputs import Poscar

def silence_deprecation_warnings():
    """
    屏蔽掉 DeprecationWarning，避免脚本中断或刷屏。
    直接调用一次即可。
    """
    def fxn():
        warnings.warn("deprecated", DeprecationWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
    warnings.filterwarnings("ignore")


def read_poscar(folder: str):
    """
    读取给定文件夹下的 POSCAR 文件，返回 pymatgen.Structure 对象。
    """
    poscar_path = folder.rstrip("/") + "/POSCAR"
    return Poscar.from_file(poscar_path).structure


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
