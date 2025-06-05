#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_700scalers.py
──────────────────────────────────────────
* 读训练 CSV (列名 files)
* 对每个 POSCAR 直接调用 fp.fp_atom(num_gamma=70, …) 生成 **700 维** 指纹
    70 × (radial 1  + dipole 3 + quad 5)  = 70 × 9 = 630
  再额外附加 70 组纯径向               ➜            + 70
* pad 到同一 padding_size
* 组装四个元素通道的矩阵 → MaxAbsScaler.fit
* 保存为 Scale700_C/H/N/O.joblib
"""

import sys, os, argparse, itertools
from pathlib import Path
import numpy as np
from tqdm import tqdm
from joblib import dump
from sklearn.preprocessing import MaxAbsScaler

# --- 把仓库根目录加到 PYTHONPATH
repo = Path(__file__).resolve().parents[2]
sys.path.append(str(repo))

from dftpy.data_io import read_file_list, get_max_atom_count
from dftpy.fp import fp_atom, pad_to

# ---------------- CLI -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train-csv", required=True, help="CSV with column 'files'")
parser.add_argument("--out-dir",   default=str(repo / "dftpy" / "scalers"),
                    help="where to save *.joblib")
parser.add_argument("--grid-spacing",   type=float, default=0.7)
parser.add_argument("--cut-off-rad",    type=float, default=5.0)
parser.add_argument("--widest-gaussian",type=float, default=6.0)
parser.add_argument("--narrowest-gaussian",type=float, default=0.5)
parser.add_argument("--num-gamma",      type=int,   default=70)   # ★ 70 → 700-dim
args = parser.parse_args()

folders = read_file_list(args.train_csv, col="files")
if not folders:
    sys.exit("❌ CSV 列表为空")

pad_size = int(get_max_atom_count(folders) * 1.0)      # 与训练时相同
print(f"Samples={len(folders)}   padding_size={pad_size}")

# holder   C / H / N / O
buf = {e: [] for e in "CHNO"}
elem_map = {6:"C", 1:"H", 7:"N", 8:"O"}                # z → symbol

for f in tqdm(folders, desc="fingerprint"):
    poscar = Path(f) / "POSCAR"
    dset, _, _, _, at_elem = fp_atom(
        poscar,
        args.grid_spacing, args.cut_off_rad,
        args.widest_gaussian, args.narrowest_gaussian,
        args.num_gamma,
    )                                                  # (n_atoms, 630)
    # ➊  extra pure-radial: 使用 radial[::9] 的那一列即可
    radial_only = dset[:, ::9]                         # shape (n_atoms, 70)
    fp700 = np.hstack([dset, radial_only])             # (n_atoms, 700)

    # ➋  pad & 分元素写入缓冲
    beg = 0
    for sym, n in zip("CHNO", at_elem):
        if n == 0:
            buf[sym].append(np.zeros((pad_size, 700), np.float32))
        else:
            part = fp700[beg:beg+n]
            buf[sym].append(pad_to(part, pad_size))
        beg += n

# ➌  组装 & fit scaler
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
for sym in "CHNO":
    X = np.vstack(buf[sym])           # (samples*P, 700)
    print(f"Fit scaler {sym}: X.shape={X.shape}")
    sc = MaxAbsScaler().fit(X)
    dump(sc, out_dir / f"Scale700_{sym}.joblib")
    print("   ✔ saved", out_dir / f"Scale700_{sym}.joblib")

print("\n✅ ALL DONE")
