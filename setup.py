# setup.py
from setuptools import setup, find_packages

setup(
    name="my_mldft",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="A PyTorch-based ML‐DFT package for charge, energy, and DOS prediction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_mldft",
    packages=find_packages(include=["mldft", "mldft.*"]),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8",
        "numpy>=1.19",
        "pandas>=1.1",
        "pymatgen>=2020.12",
        "scikit-learn>=0.24",
        "h5py>=2.10",
        "tqdm>=4.50"
    ],
    entry_points={
        "console_scripts": [
            "mldft=mldft.cli:main"
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

