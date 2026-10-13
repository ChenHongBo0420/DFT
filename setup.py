"""Package metadata; install this checkout with python -m pip install -e ."""

from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

setup(
    name="DFTpy",
    version="0.1.0",
    author="Chen Hongbo",
    author_email="chenjushua@gmail.com",
    description="PyTorch research code for charge, energy, force, stress and DOS prediction",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/ChenHongBo0420/DFT",
    packages=find_packages(include=["dftpy", "dftpy.*"]),
    package_data={"dftpy": ["scalers/*.joblib", "scalers/README.md"]},
    python_requires=">=3.10",
    install_requires=["torch>=2.6", "numpy", "pymatgen", "scikit-learn", "joblib", "h5py", "tqdm"],
    extras_require={"plot": ["matplotlib"], "notebooks": ["jupyterlab", "matplotlib"]},
    entry_points={"console_scripts": ["dftpy=dftpy.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
