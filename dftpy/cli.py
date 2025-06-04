# dftpy/cli.py
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from pymatgen.io.vasp.inputs import Poscar  # only used for type hints / IDE help

from .utils import silence_deprecation_warnings, read_poscar
from .data_io import (
    get_max_atom_count,
    read_file_list,
    save_charges,
    save_dos,
    save_energy,
)
from .chg import infer_charges, load_pretrained_chg_model, train_chg_model
from .dos import infer_dos, load_pretrained_dos_model, train_dos_model
from .energy import (
    infer_energy,
    load_pretrained_energy_model,
    train_energy_model,
)

# -----------------------------------------------------------------------------
# NEW: robust helper to locate POSCAR file no matter what path we receive
# -----------------------------------------------------------------------------

def _find_poscar(folder: str | Path) -> tuple[Path, Path]:
    """Return (poscar_path, base_dir).

    The CSV entry might be:
    1. structure root – e.g. …/R1000003
    2. “POSCAR” directory – e.g. …/R1000003/POSCAR
    3. POSCAR file – e.g. …/R1000003/POSCAR or …/R1000003/POSCAR/POSCAR

    We scan in the following order and stop at the first file hit:
      • <folder>               (if file and name == POSCAR)
      • <folder>/POSCAR        (file)
      • <folder>/POSCAR/POSCAR (file)

    base_dir is chosen as the *directory where we will store Coef_*.npy*.
    Here we keep it simple: use the **directory that directly contains
    the POSCAR file** – this works for all training / inference paths.
    """

    f = Path(folder)
    candidates: list[Path] = []

    # Case 1: user passed the POSCAR file directly
    if f.is_file() and f.name.upper() == "POSCAR":
        candidates.append(f)
    else:
        # Case 2 & 3: structure dir or POSCAR dir
        candidates.extend([
            f / "POSCAR",
            f / "POSCAR" / "POSCAR",
        ])

    for c in candidates:
        if c.is_file():
            return c, c.parent  # poscar_path, base_dir

    raise FileNotFoundError(f"在路径 {folder} 及其下级目录中未找到 POSCAR 文件")


# -----------------------------------------------------------------------------
# Patched: save charge‑coefficients as Coef_*.npy (C/H/N/O) per structure
# -----------------------------------------------------------------------------

def _save_coef_npy_for_folder(
        folder: str,
        chg_model: torch.nn.Module,
        padding_size: int,
        args):
    """对单个 structure 生成 Coef_C/H/N/O.npy。"""

    # 1️⃣ 找到真实 POSCAR 文件
    poscar_path, base_dir = _find_poscar(folder)

    # 2️⃣ 推理得到 “每原子系数矩阵”
    all_coef = infer_charges(str(poscar_path), chg_model, padding_size, args)
    if isinstance(all_coef, torch.Tensor):
        all_coef = all_coef.detach().cpu().numpy()

    # —— 若 infer_charges 只返回一维 (N,) 电荷值 → 升成 (N,1)
    if all_coef.ndim == 1:
        all_coef = all_coef.reshape(-1, 1)

    # 3️⃣ 统计各元素原子数
    struct = Poscar.from_file(str(poscar_path)).structure
    at_C = struct.species.count("C")
    at_H = struct.species.count("H")
    at_N = struct.species.count("N")
    at_O = struct.species.count("O")
    total_atoms = at_C + at_H + at_N + at_O

    if all_coef.shape[0] != total_atoms:
        raise ValueError(
            f"{base_dir}: 系数行数 {all_coef.shape[0]} ≠ 原子总数 {total_atoms}"
        )

    # 4️⃣ 拆分 & 保存
    i1 = at_C
    i2 = i1 + at_H
    i3 = i2 + at_N
    coef_split = {
        "C": all_coef[0:i1, :],
        "H": all_coef[i1:i2, :],
        "N": all_coef[i2:i3, :],
        "O": all_coef[i3:total_atoms, :],
    }

    for elem, mat in coef_split.items():
        # fp_chg_norm 期望 1-D (P,)；若只有 1 列则压扁
        if mat.ndim == 2 and mat.shape[1] == 1:
            mat_to_save = mat[:, 0].astype(np.float32)
        else:
            mat_to_save = mat.astype(np.float32)
        np.save(str(base_dir / f"Coef_{elem}.npy"), mat_to_save)

def parse_args():
    parser = argparse.ArgumentParser(prog="dftpy", description="PyTorch-based ML-DFT: charge, energy, DOS")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ========= “train” 子命令 ==========
    train_parser = subparsers.add_parser("train", help="Train models: chg/energy/dos/all")
    train_parser.add_argument("--task", required=True, choices=["chg", "energy", "dos", "all"])
    train_parser.add_argument("--train-list", dest="train_list", type=str, required=True,
                              help="Train CSV, 列名为 files")
    train_parser.add_argument("--val-list", dest="val_list", type=str, required=True,
                              help="Val   CSV, 列名为 files")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    train_parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--grid-spacing", dest="grid_spacing", type=float, default=1.0)
    train_parser.add_argument("--cut-off-rad", dest="cut_off_rad", type=float, default=6.0)
    train_parser.add_argument("--widest-gaussian", dest="widest_gaussian", type=float, default=2.0)
    train_parser.add_argument("--narrowest-gaussian", dest="narrowest_gaussian", type=float, default=0.5)
    train_parser.add_argument("--num-gamma", dest="num_gamma", type=int, default=10)
    train_parser.add_argument("--padding-multiplier", dest="padding_multiplier", type=float, default=1.0)

    # ========= “infer” 子命令 ==========
    infer_parser = subparsers.add_parser("infer", help="Infer on new structures")
    infer_parser.add_argument("--infer-list", dest="infer_list", type=str, required=True,
                              help="Predict CSV, 列名为 file_loc_test")
    infer_parser.add_argument("--output-dir", dest="output_dir", type=str, default="results")
    infer_parser.add_argument("--predict-chg", dest="predict_chg", action="store_true")
    infer_parser.add_argument("--predict-energy", dest="predict_energy", action="store_true")
    infer_parser.add_argument("--predict-dos", dest="predict_dos", action="store_true")
    infer_parser.add_argument("--grid-spacing", dest="grid_spacing", type=float, default=1.0)
    infer_parser.add_argument("--cut-off-rad", dest="cut_off_rad", type=float, default=6.0)
    infer_parser.add_argument("--widest-gaussian", dest="widest_gaussian", type=float, default=2.0)
    infer_parser.add_argument("--narrowest-gaussian", dest="narrowest_gaussian", type=float, default=0.5)
    infer_parser.add_argument("--num-gamma", dest="num_gamma", type=int, default=10)
    infer_parser.add_argument("--padding-multiplier", dest="padding_multiplier", type=float, default=1.0)
    infer_parser.add_argument("--plot-dos", dest="plot_dos", action="store_true")

    return parser.parse_args()


def main():
    silence_deprecation_warnings()
    args = parse_args()

    if args.mode == "train":
        # ─── 1) 读取 CSV，得到文件夹列表 ───
        train_folders = read_file_list(args.train_list, col="files")
        val_folders = read_file_list(args.val_list, col="files")

        # ─── 2) 计算 padding_size: max_atom_count * padding_multiplier ───
        max_train = get_max_atom_count(train_folders)
        max_val = get_max_atom_count(val_folders)
        padding_size = int(max(max_train, max_val) * args.padding_multiplier)
        if padding_size < 1:
            raise ValueError("Computed padding_size < 1; 请检查输入文件或 padding_multiplier。")

        # ─── 3) 依次训练 ───
        if args.task in ("chg", "all"):
            print("[INFO] 训练电荷模型 …")
            train_chg_model(train_folders, val_folders, padding_size, args)

        if args.task in ("energy", "all"):
            print("[INFO] 训练能量模型 …")
            # 必须先加载电荷模型
            chg_ckpt = "best_chg.pth"
            try:
                chg_model = load_pretrained_chg_model(chg_ckpt, padding_size)
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载电荷模型权重: {e}")
                return
            train_energy_model(train_folders, val_folders, chg_model, padding_size, args)

        if args.task in ("dos", "all"):
            print("[INFO] 训练 DOS 模型 …")
            # 1) 先加载电荷模型
            chg_ckpt = "best_chg.pth"
            try:
                chg_model = load_pretrained_chg_model(chg_ckpt, padding_size)
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载电荷模型权重: {e}")
                return

            # 2) → 【新增】为 train_folders 和 val_folders 分别生成 Coef_*.npy，写回各自文件夹
            print("[INFO] 生成并保存 Coef_*.npy 数据，以备 DOS 训练使用 …")
            for folder in train_folders + val_folders:
                _save_coef_npy_for_folder(folder, chg_model, padding_size, args)

            # 3) 最后再调用 train_dos_model（dos.py 会去各个 folder 里找 Coef_*.npy）
            train_dos_model(train_folders, val_folders, padding_size, args)

    elif args.mode == "infer":
        # ─── 1) 读取 CSV，得到要预测的文件夹列表 ───
        infer_folders = read_file_list(args.infer_list, col="file_loc_test")
        if not infer_folders:
            print("[WARN] infer-list 为空，没有任何文件夹可预测。")
            return

        os.makedirs(args.output_dir, exist_ok=True)

        # 计算 padding_size 以供所有模型使用
        max_count = get_max_atom_count(infer_folders)
        padding_size = int(max_count * args.padding_multiplier)
        if padding_size < 1:
            raise ValueError("Computed padding_size < 1; 请检查输入文件或 padding_multiplier。")

        chg_model = None
        energy_model = None
        dos_model = None

        # ─── 2) 加载电荷模型（如需要） ───
        if args.predict_chg:
            chg_ckpt = "best_chg.pth"
            try:
                chg_model = load_pretrained_chg_model(chg_ckpt, padding_size)
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载电荷模型权重: {e}")
                return

        # ─── 3) 加载能量模型（如需） ───
        if args.predict_energy:
            energy_ckpt = "newEmodel.pth"
            fingerprint_dim = 360
            basis_dim = 9
            dim_C = fingerprint_dim + basis_dim
            dim_H = fingerprint_dim + basis_dim
            dim_N = fingerprint_dim + basis_dim
            dim_O = fingerprint_dim + basis_dim

            try:
                energy_model = load_pretrained_energy_model(
                    energy_ckpt,
                    padding_size=int(get_max_atom_count(infer_folders) * args.padding_multiplier),
                    dim_C=dim_C, dim_H=dim_H, dim_N=dim_N, dim_O=dim_O
                )
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载能量模型权重: {e}")
                return

        # ─── 4) 加载 DOS 模型（如需） ───
        if args.predict_dos:
            dos_ckpt = "best_dos.pth"
            try:
                dos_model = load_pretrained_dos_model(
                    dos_ckpt,
                    padding_size=int(get_max_atom_count(infer_folders) * args.padding_multiplier)
                )
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载 DOS 模型权重: {e}")
                return

        # ─── 5) 循环对每个结构做预测并保存结果 ───
        for folder in infer_folders:
            print(f"[INFO] 处理 {folder} …")
            pdbasename = os.path.basename(folder.rstrip("/"))

            # 【电荷】
            if args.predict_chg:
                try:
                    chg_vals = infer_charges(
                        folder,
                        chg_model,
                        int(get_max_atom_count(infer_folders) * args.padding_multiplier),
                        args
                    )
                    save_charges(chg_vals, os.path.join(args.output_dir, f"charges_{pdbasename}.txt"))
                except Exception as e:
                    print(f"[WARN] 电荷预测失败 ({folder}): {e}")

            # 【能量】
            if args.predict_energy:
                try:
                    e_val, forces, stress = infer_energy(
                        folder,
                        chg_model,
                        energy_model,
                        int(get_max_atom_count(infer_folders) * args.padding_multiplier),
                        args
                    )
                    save_energy(e_val, forces, stress, os.path.join(args.output_dir, f"energy_{pdbasename}.txt"))
                except Exception as e:
                    print(f"[WARN] 能量预测失败 ({folder}): {e}")

            # 【DOS】
            if args.predict_dos:
                try:
                    energy_grid, dos_curve, vb, cb, bg, uncertainty = infer_dos(
                        folder,
                        chg_model,
                        dos_model,
                        int(get_max_atom_count(infer_folders) * args.padding_multiplier),
                        args
                    )
                    save_dos(
                        energy_grid, dos_curve, vb, cb, bg,
                        os.path.join(args.output_dir, f"dos_{pdbasename}.txt")
                    )
                    if args.plot_dos:
                        # 可选：如果在 utils 里实现了 plot_dos，就在此调用
                        # from .utils import plot_dos
                        # plot_dos(energy_grid, dos_curve, vb, cb, os.path.join(args.output_dir, f"dos_plot_{pdbasename}.png"))
                        pass
                except Exception as e:
                    print(f"[WARN] DOS 预测失败 ({folder}): {e}")

    else:
        raise ValueError("Unknown mode. Choose from 'train' or 'infer'.")


if __name__ == "__main__":
    main()
