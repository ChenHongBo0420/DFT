#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_700scalers.py
用 700-维指纹为 C/H/N/O 分别拟合 MaxAbsScaler
-----------------------------------------------------------------
USAGE
  python -u dftpy/scripts/generate_700scalers.py \
         --train-csv  /path/to/Train.csv \
         --out-dir    dftpy/scalers
"""

import argparse, os, sys
from pathlib import Path
import joblib, numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler

# ─────────────────── CLI ───────────────────
p = argparse.ArgumentParser()
p.add_argument('--train-csv', required=True,
               help='CSV 必须有一列名为 files，指向结构文件夹')
p.add_argument('--out-dir',   default='dftpy/scalers',
               help='保存 *.joblib 的目录（相对项目根）')
args = p.parse_args()

# ───── 确保 “项目根” 在 sys.path ─────
repo = Path(__file__).resolve().parents[2]     # /content/DFT
if repo.as_posix() not in sys.path:
    sys.path.insert(0, repo.as_posix())        # 现在才能 import dftpy.*
# ────────────────────────────────────────────
from dftpy.data_io import (
    read_file_list, get_max_atom_count,
    get_efp_data, pad_efp_data
)

# ① 读训练列表 & padding_size ---------------------------------------------------
folders = read_file_list(args.train_csv, col='files')
pad = int(get_max_atom_count(folders))
print(f'Samples={len(folders):d}   padding_size={pad}')

# ② 取 700-维指纹 --------------------------------------------------------------
_, _, _, fp_list, basis_list, _, _, elem_list = get_efp_data(folders)

# pad_efp_data 把 700-维指纹拆成四块元素矩阵 X_1..X_4
_,_,_,_, X_C, X_H, X_N, X_O, *_ = pad_efp_data(
    elem_list, fp_list, None, basis_list, pad
)

# ③ 拍平成 (N*pad, 700) 后 fit --------------------------------------------------
def fit_block(X, tag):
    flat = X.reshape(-1, X.shape[-1])
    print(f'Fit scaler {tag}: X.shape={flat.shape}')
    sc = MaxAbsScaler().fit(flat)
    out = (repo/args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(sc, out/f'Scale700_{tag}.joblib')
    print('  ✔ saved', out/f'Scale700_{tag}.joblib')

fit_block(X_C, 'C')
fit_block(X_H, 'H')
fit_block(X_N, 'N')
fit_block(X_O, 'O')

print('\n✅ ALL DONE')
