# setup.py

from setuptools import setup, find_packages
import pathlib

# 项目根目录
here = pathlib.Path(__file__).parent.resolve()

# 读取 README.md 作为 long_description
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="DFTpy", 
    version="0.1.0",
    author="Chen Hongbo",
    author_email="chenjushua@gmail.com",
    description="A PyTorch‐based ML‐DFT package for charge, energy and DOS prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChenHongBo0420/DFT",
    packages=find_packages(exclude=["tests", "docs"]),  # 自动打包所有子模块
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8",          # PyTorch
        "numpy>=1.19",         # 数值计算
        "pandas>=1.1",         # 表格读取/处理
        "pymatgen>=2020.12",   # POSCAR 解析等
        "scikit-learn>=0.24",  # 指标计算
        "h5py>=2.10",          # HDF5 读写
        "tqdm>=4.50"           # 进度条
    ],
    entry_points={
        "console_scripts": [
            # 安装后可以执行 `dftpy train` 或 `dftpy infer`
            "dftpy=dftpy.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
