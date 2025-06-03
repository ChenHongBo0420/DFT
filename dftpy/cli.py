# dftpy/cli.py

import argparse
import os
import torch
import numpy as np    # 新增，用来保存 .npy
from pathlib import Path

from .utils import silence_deprecation_warnings, read_poscar
from .data_io import read_file_list, get_max_atom_count, save_charges, save_energy, save_dos
from .chg import train_chg_model, load_pretrained_chg_model, infer_charges
from .energy import train_energy_model, load_pretrained_energy_model, infer_energy
from .dos import train_dos_model, load_pretrained_dos_model, infer_dos

# -----------------------------------------------------------------------------
# 新增：用于将推理出的电荷系数拆分并保存为 Coef_C.npy、Coef_H.npy、Coef_N.npy、Coef_O.npy
# -----------------------------------------------------------------------------
def _save_coef_npy_for_folder(folder: str, chg_model: torch.nn.Module, padding_size: int, args):
    """
    对单个结构文件夹 folder：调用 infer_charges 得到所有原子电荷系数，
    并根据元素顺序拆分成 Coef_C.npy / Coef_H.npy / Coef_N.npy / Coef_O.npy。
    最终把这四个 .npy 文件写回 folder 目录下，供后续 dos.py 使用。
    """
    # 1) 调用 infer_charges，得到一个 “所有原子系数矩阵”。
    #    返回值假设形状是 (num_total_atoms, feat_dim)；后面我们再按 “元素数量” 拆开。
    all_coef = infer_charges(folder, chg_model, padding_size, args)
    # all_coef: numpy.ndarray 或者 torch.Tensor，形状 (N_total, feat_dim)。我们统一转 np.ndarray
    if isinstance(all_coef, torch.Tensor):
        all_coef = all_coef.cpu().numpy()

    # 2) 从 POSCAR 中提取“每个元素原子个数（at_elem）”，以便把 all_coef 拆分成四份
    from .fp import fp_atom
    from .chg import chg_data

    # 2.1 先读 POSCAR：
    #     这里只需要元素顺序和个数（C,H,N,O 顺序），让后面分配每块系数
    struct = read_poscar(os.path.join(folder, "POSCAR"))
    # read_poscar 应该返回一个 pymatgen Structure 对象
    # 也可以直接用 pymatgen.get_structure_from_file(folder+'/POSCAR')，效果类似
    # 以下手动统计 C/H/N/O 数量
    elem_dict = {"C": 0, "H": 0, "N": 0, "O": 0}
    for site in struct:
        s = site.specie.symbol
        if s in elem_dict:
            elem_dict[s] += 1
    at_C, at_H, at_N, at_O = elem_dict["C"], elem_dict["H"], elem_dict["N"], elem_dict["O"]

    # 3) 现在把 all_coef 按 [C个数, H个数, N个数, O个数] 依次切片
    #    假设 all_coef 顺序正好对应 “先所有 C，再所有 H，再所有 N，再所有 O”。
    #    如果 infer_charges 的返回顺序与元素排列一致，这样拆就没问题。
    #    如果顺序不一样，需要你根据 chg_data 的输出或实际情况来调整下面的切片逻辑。
    i1 = at_C
    i2 = at_C + at_H
    i3 = at_C + at_H + at_N
    i4 = at_C + at_H + at_N + at_O   # 总原子数

    # 让 all_coef shape = (i4, feat_dim)，并把索引 [0:i1], [i1:i2], [i2:i3], [i3:i4] 分别写入各元素
    Coef_C = all_coef[0:i1, :].astype(np.float32) if i1 > 0 else np.zeros((0, all_coef.shape[1]), dtype=np.float32)
    Coef_H = all_coef[i1:i2, :].astype(np.float32) if at_H > 0 else np.zeros((0, all_coef.shape[1]), dtype=np.float32)
    Coef_N = all_coef[i2:i3, :].astype(np.float32) if at_N > 0 else np.zeros((0, all_coef.shape[1]), dtype=np.float32)
    Coef_O = all_coef[i3:i4, :].astype(np.float32) if at_O > 0 else np.zeros((0, all_coef.shape[1]), dtype=np.float32)

    # 4) 保存到 folder 目录下
    np.save(os.path.join(folder, "Coef_C.npy"), Coef_C)
    np.save(os.path.join(folder, "Coef_H.npy"), Coef_H)
    np.save(os.path.join(folder, "Coef_N.npy"), Coef_N)
    np.save(os.path.join(folder, "Coef_O.npy"), Coef_O)


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
            # 1) 先加载电荷模型（必须有 best_chg.pth）
            chg_ckpt = "best_chg.pth"
            try:
                chg_model = load_pretrained_chg_model(chg_ckpt, padding_size)
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载电荷模型权重: {e}")
                return

            # 2) → 【新增】对 train_folders 和 val_folders 分别生成 Coef_*.npy，保存到各自文件夹
            print("[INFO] 生成并保存 Coef_*.npy 数据，以备 DOS 训练使用 …")
            for folder in train_folders + val_folders:
                _save_coef_npy_for_folder(folder, chg_model, padding_size, args)

            # 3) 最后再调用 train_dos_model，dos.py 会去各个 folder 里找 Coef_C.npy 等
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
