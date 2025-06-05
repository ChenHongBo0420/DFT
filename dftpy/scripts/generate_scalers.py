#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_700scalers.py
给 C/H/N/O 四类 700-D 指纹各自 fit MaxAbsScaler，并保存为
    dftpy/scalers/Scale700_{C,H,N,O}.joblib
用法：
    python dftpy/scripts/generate_700scalers.py \
           --train-csv /path/to/Train.csv \
           [--padding-mult 1.0]
"""

import os, sys, argparse
from pathlib import Path
import joblib, numpy as np
from sklearn.preprocessing import MaxAbsScaler

# ─── 1. CLI ───────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--train-csv", required=True,
                help="CSV 必须有列 files，指向结构文件夹")
ap.add_argument("--padding-mult", type=float, default=1.0,
                help="padding_multiplier，与正式训练保持一致")
args = ap.parse_args()

# ─── 2. 把项目根目录加入 PYTHONPATH ─────────────────────────────
repo = Path(__file__).resolve().parents[1]        # …/DFT/
sys.path.append(str(repo))
from dftpy.data_io import read_file_list, get_max_atom_count, get_efp_data, pad_efp_data

sc_dir = repo / "dftpy" / "scalers"; sc_dir.mkdir(parents=True, exist_ok=True)

# ─── 3. 读取样本 & padding_size ────────────────────────────────
folders = read_file_list(args.train_csv, col="files")
if not folders:
    sys.exit("❌ CSV 为空？")

pad_size = int(get_max_atom_count(folders) * args.padding_mult)
print(f"Samples={len(folders)}   padding_size={pad_size}", flush=True)

# ─── 4. 生成 EFP 数据，并 pad 成 X_1..X_4 ─────────────────────
ener_ref, forces_pre, press_ref, X_pre, basis_pre, \
    at_list, el_list, elem_list = get_efp_data(folders)

data = pad_efp_data(elem_list, X_pre, forces_pre, basis_pre, pad_size)
_, _, _, _, X1, X2, X3, X4, *_ = data      # 只要 X1..X4

# ─── 5. 取前 700 个特征列 ─────────────────────────────────────
def first700(X):
    if X.shape[-1] < 700:
        raise ValueError(f"feat_dim={X.shape[-1]} < 700，会越界")
    return X[..., :700]

X_C, X_H, X_N, X_O = map(first700, (X1, X2, X3, X4))

# ─── 6. reshape → fit scaler ─────────────────────────────────
def flat(X):
    n, p, d = X.shape
    return X.reshape(n*p, d)               # (N*P, 700)

pairs = zip("CHNO", (X_C, X_H, X_N, X_O))
for elem, mat in pairs:
    print(f"Fit scaler {elem}: X.shape={mat.shape}", flush=True)
    sc = MaxAbsScaler().fit(flat(mat))
    out = sc_dir / f"Scale700_{elem}.joblib"
    joblib.dump(sc, out)
    print("   ✔ saved", out, flush=True)

print("✅ ALL DONE", flush=True)
