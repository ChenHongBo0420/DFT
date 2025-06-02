# dftpy/cli.py

import argparse
import os
import torch
from .utils import silence_deprecation_warnings, read_poscar
from .data_io import read_file_list, get_max_atom_count, save_charges, save_energy, save_dos
from .chg import train_chg_model, load_pretrained_chg_model, infer_charges
from .energy import train_energy_model, load_pretrained_energy_model, infer_energy
from .dos import train_dos_model, load_pretrained_dos_model, infer_dos

def parse_args():
    parser = argparse.ArgumentParser(prog="dftpy", description="PyTorch-based ML-DFT: charge, energy, DOS")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # “train” 子命令
    train_parser = subparsers.add_parser("train", help="Train models: chg/energy/dos/all")
    train_parser.add_argument("--task", required=True, choices=["chg","energy","dos","all"])
    train_parser.add_argument("--train-list", type=str, required=True, help="Train CSV, 列名为 files")
    train_parser.add_argument("--val-list", type=str, required=True, help="Val   CSV, 列名为 files")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--grid-spacing", type=float, default=1.0)
    train_parser.add_argument("--cut-off-rad", type=float, default=6.0)
    train_parser.add_argument("--widest-gaussian", type=float, default=2.0)
    train_parser.add_argument("--narrowest-gaussian", type=float, default=0.5)
    train_parser.add_argument("--num-gamma", type=int, default=10)
    train_parser.add_argument("--padding-multiplier", type=float, default=1.0)

    # “infer” 子命令
    infer_parser = subparsers.add_parser("infer", help="Infer on new structures")
    infer_parser.add_argument("--infer-list", type=str, required=True, help="Predict CSV, 列名为 file_loc_test")
    infer_parser.add_argument("--output-dir", type=str, default="results")
    infer_parser.add_argument("--predict-chg", action="store_true")
    infer_parser.add_argument("--predict-energy", action="store_true")
    infer_parser.add_argument("--predict-dos", action="store_true")
    infer_parser.add_argument("--grid-spacing", type=float, default=1.0)
    infer_parser.add_argument("--cut-off-rad", type=float, default=6.0)
    infer_parser.add_argument("--widest-gaussian", type=float, default=2.0)
    infer_parser.add_argument("--narrowest-gaussian", type=float, default=0.5)
    infer_parser.add_argument("--num-gamma", type=int, default=10)
    infer_parser.add_argument("--padding-multiplier", type=float, default=1.0)
    infer_parser.add_argument("--plot-dos", action="store_true")

    return parser.parse_args()


def main():
    silence_deprecation_warnings()
    args = parse_args()

    if args.mode == "train":
        # 1) 读取 CSV，得到文件夹列表
        train_folders = read_file_list(args.train_list, col="files")
        val_folders = read_file_list(args.val_list, col="files")
        # 2) 计算 padding_size：倍数 * max_atom_count
        max_train = get_max_atom_count(train_folders)
        max_val = get_max_atom_count(val_folders)
        padding_size = int(max(max_train, max_val) * args.padding_multiplier)

        # 3) 按 task 依次训练
        if args.task in ("chg", "all"):
            print("[INFO] 训练电荷模型 …")
            train_chg_model(train_folders, val_folders, padding_size, args)

        if args.task in ("energy", "all"):
            print("[INFO] 训练能量模型 …")
            # 先加载电荷模型的 checkpoint
            chg_ckpt = "best_chg.pth"  # 你可以改为从 args 中读
            chg_model = load_pretrained_chg_model(chg_ckpt, padding_size)
            train_energy_model(train_folders, val_folders, chg_model, padding_size, args)

        if args.task in ("dos", "all"):
            print("[INFO] 训练 DOS 模型 …")
            chg_ckpt = "best_chg.pth"
            dos_ckpt_pre = None  # 如果 DOS 训练不需要能量模型的 checkpoint，可设为 None
            chg_model = load_pretrained_chg_model(chg_ckpt, padding_size)
            train_dos_model(train_folders, val_folders, chg_model, dos_ckpt_pre, padding_size, args)

    elif args.mode == "infer":
        # 1) 读取 CSV，得到要预测的文件夹列表
        infer_folders = read_file_list(args.infer_list, col="file_loc_test")
        os.makedirs(args.output_dir, exist_ok=True)
        # 2) 先加载所需模型
        if args.predict_chg:
            chg_ckpt = "best_chg.pth"
            chg_model = load_pretrained_chg_model(chg_ckpt, padding_size=int(get_max_atom_count(infer_folders)*args.padding_multiplier))
        if args.predict_energy:
            energy_ckpt = "best_energy.pth"
            energy_model = load_pretrained_energy_model(energy_ckpt, padding_size=int(get_max_atom_count(infer_folders)*args.padding_multiplier))
        if args.predict_dos:
            dos_ckpt = "best_dos.pth"
            dos_model = load_pretrained_dos_model(dos_ckpt, padding_size=int(get_max_atom_count(infer_folders)*args.padding_multiplier))

        # 3) 循环每个 target folder，逐次预测并保存
        for folder in infer_folders:
            print(f"[INFO] 处理 {folder} …")
            pdbasename = os.path.basename(folder.rstrip("/"))
            # 【电荷】
            if args.predict_chg:
                chg_vals = infer_charges(folder, chg_model, int(get_max_atom_count(infer_folders)*args.padding_multiplier), args)
                save_charges(chg_vals, os.path.join(args.output_dir, f"charges_{pdbasename}.txt"))
            # 【能量】
            if args.predict_energy:
                e_val, forces, stress = infer_energy(folder, chg_model, energy_model, int(get_max_atom_count(infer_folders)*args.padding_multiplier), args)
                save_energy(e_val, forces, stress, os.path.join(args.output_dir, f"energy_{pdbasename}.txt"))
            # 【DOS】
            if args.predict_dos:
                energy_grid, dos_curve, vb, cb, bg, uncertainty = infer_dos(folder, chg_model, dos_model, int(get_max_atom_count(infer_folders)*args.padding_multiplier), args)
                save_dos(energy_grid, dos_curve, vb, cb, bg, os.path.join(args.output_dir, f"dos_{pdbasename}.txt"))
                if args.plot_dos:
                    from .utils import save_stdout_to_file
                    # 也可以在这里调用一个画图函数，但如果不需要可省略
                    # e.g., plot_dos(energy_grid, dos_curve, vb, cb, os.path.join(args.output_dir, f"dos_plot_{pdbasename}.png"))

    else:
        raise ValueError("Unknown mode. Choose from 'train' or 'infer'.")


if __name__ == "__main__":
    main()
