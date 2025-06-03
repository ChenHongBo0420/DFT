# dftpy/cli.py

import argparse
import os
import torch
import numpy as np
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
    对单个“folder”：
    - folder 可能是一个“目录”（目录里有 POSCAR），
      也可能直接是一个 POSCAR 文件路径。
    本函数会先定位到真正的 POSCAR 文件，然后调用 infer_charges 算出所有原子电荷系数，
    接着根据结构里 C/H/N/O 的原子数目，把所有系数矩阵按元素依次切分，
    并分别保存为 folder/Coef_C.npy、folder/Coef_H.npy、folder/Coef_N.npy、folder/Coef_O.npy。
    """
    p = Path(folder)
    # 1) 定位到 POSCAR 文件
    if p.is_file() and p.name.upper() == "POSCAR":
        poscar_path = p
        base_dir = p.parent
    else:
        poscar_path = p / "POSCAR"
        base_dir = p
    if not poscar_path.is_file():
        raise FileNotFoundError(f"找不到 POSCAR 文件：{poscar_path}")

    # 2) 用 infer_charges 得到 “所有原子电荷系数” — NumPy 数组 or Tensor
    #    注意：infer_charges 返回的是“二维 NumPy 数组”：shape=(N_total_atoms, )
    #    但我们在之前改过的 infer_charges 中，会返回一个一维 array “全原子序列”的电荷值列表。
    #    为了训练 DOS，我们需要“每个原子的 exp/coef 特征矩阵”，不是直接的电荷值。
    #    这里假设 infer_charges 已修改为直接返回“每个原子 exp/coef 矩阵”（shape = (N_total_atoms, feat_dim)）。
    #
    #    如果你当前的 infer_charges 还是返回一维电荷列表，那么你需要另行调整逻辑：
    #    ——本示例假设 infer_charges 已经按 “Chg->C,H,N,O 四支通路的 exp/coef” 直接输出矩阵，
    #      如果不是，需要把推理接口改为输出那 4 个元素的系数矩阵（见前面解答）。
    #
    #    下面示例以 infer_charges 返回 (N_total_atoms, feat_dim) 为前提。
    all_coef = infer_charges(str(poscar_path), chg_model, padding_size, args)
    # 如果 infer_charges 返回的是 Tensor，需要先 detach, cpu, numpy
    if isinstance(all_coef, torch.Tensor):
        all_coef = all_coef.detach().cpu().numpy()

    # 3) 读取 POSCAR 得到结构，统计 C/H/N/O 原子数目
    struct = read_poscar(str(poscar_path))
    elem_counts = [struct.species.count(e) for e in ("C", "H", "N", "O")]
    at_C, at_H, at_N, at_O = elem_counts
    total_atoms = at_C + at_H + at_N + at_O

    # 4) 检查 all_coef 行数是否等于 total_atoms
    if all_coef.shape[0] != total_atoms:
        raise ValueError(
            f"在文件夹 {base_dir} 中，infer_charges 返回的系数行数 {all_coef.shape[0]} "
            f"与结构实际原子数 {total_atoms} 不一致。"
        )

    # 5) 把 all_coef 按 [C, H, N, O] 拆分
    i1 = at_C
    i2 = at_C + at_H
    i3 = at_C + at_H + at_N
    i4 = total_atoms

    Coef_C = all_coef[0:i1, :].astype(np.float32) if at_C > 0 else np.zeros((0, all_coef.shape[1]), dtype=np.float32)
    Coef_H = all_coef[i1:i2, :].astype(np.float32) if at_H > 0 else np.zeros((0, all_coef.shape[1]), dtype=np.float32)
    Coef_N = all_coef[i2:i3, :].astype(np.float32) if at_N > 0 else np.zeros((0, all_coef.shape[1]), dtype=np.float32)
    Coef_O = all_coef[i3:i4, :].astype(np.float32) if at_O > 0 else np.zeros((0, all_coef.shape[1]), dtype=np.float32)

    # 6) 将这四个矩阵分别保存到各自文件夹
    np.save(str(base_dir / "Coef_C.npy"), Coef_C)
    np.save(str(base_dir / "Coef_H.npy"), Coef_H)
    np.save(str(base_dir / "Coef_N.npy"), Coef_N)
    np.save(str(base_dir / "Coef_O.npy"), Coef_O)


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
