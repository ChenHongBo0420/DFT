#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_700scalers.py
--------------------------------------------------------------
1) 读取训练 CSV（列名: files）获得所有样本文件夹
2) 对每个样本的 POSCAR 计算 **700 维指纹**（num_gamma=70）
3) 按元素 (C/H/N/O) 累积 → fit 4 个 MaxAbsScaler
4) 保存为
      dftpy/scalers/Scale700_C.joblib
      dftpy/scalers/Scale700_H.joblib
      dftpy/scalers/Scale700_N.joblib
      dftpy/scalers/Scale700_O.joblib
--------------------------------------------------------------
"""

import os, sys, joblib, numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler

# ╭───────────────────────── 手动修改区 ─────────────────────────╮
train_list_csv = "/content/drive/MyDrive/DFT_CSVs/Train.csv"   # ← 训练 CSV
grid_spacing       = 0.7
cut_off_rad        = 5.0
widest_gaussian    = 6.0
narrowest_gaussian = 0.5
num_gamma          = 70        # ★ 700 维指纹
padding_multiplier = 1.0
# ╰─────────────────────────────────────────────────────────────╯

# ====== 把项目根目录加入 sys.path（假设脚本位于 dftpy/scripts/） ====
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from dftpy.data_io import read_file_list, get_max_atom_count
from dftpy.fp       import fp_atom, pad_to

# ---------- Step-0  准备 ----------
folders = read_file_list(train_list_csv, col="files")
if not folders:
    raise RuntimeError("训练 CSV 为空？")

pad_size = int(get_max_atom_count(folders) * padding_multiplier)
print(f"[INFO] 样本数={len(folders)}   padding_size={pad_size}")

buf = {e: [] for e in "CHNO"}         # 收集四个元素的 700-D 向量

# ---------- Step-1  逐结构计算 700-D 指纹 ----------
for fld in tqdm(folders, desc="fingerprint"):
    poscar = Path(fld) / "POSCAR"
    if not poscar.is_file():
        print("[SKIP] no POSCAR :", fld)
        continue

    fp700, _, _, _, _ = fp_atom(
        poscar,
        grid_spacing, cut_off_rad,
        widest_gaussian, narrowest_gaussian, num_gamma
    )                                   # (N_atoms, 700)

    # element split ------------------------------------------------------
    from pymatgen.io.vasp.inputs import Poscar
    struct = Poscar.from_file(poscar).structure
    elem_map = {6: "C", 1: "H", 7: "N", 8: "O"}

    for i, site in enumerate(struct):
        sym = elem_map.get(site.atomic_number)
        if not sym:          # ignore other elements
            continue
        vec = pad_to(fp700[i:i+1], pad_size)[0]   # (700,) after pad
        buf[sym].append(vec)

# ---------- Step-2  fit scaler & 保存 ---------------------------
sc_dir = repo_root / "dftpy" / "scalers"
sc_dir.mkdir(exist_ok=True, parents=True)

for e in "CHNO":
    X = np.vstack(buf[e])                # (atoms, 700)
    if X.size == 0:
        print(f"[WARN] 元素 {e} 在训练集缺失，跳过 scaler")
        continue
    scaler = MaxAbsScaler().fit(X)
    out = sc_dir / f"Scale700_{e}.joblib"
    joblib.dump(scaler, out)
    print(f"✔ {out.name}  saved, X.shape={X.shape}")

print("\n🚀  All done.")
