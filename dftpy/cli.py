# dftpy/cli.py

import argparse
import os
import warnings
from .data_io import read_file_list, get_max_atom_count, save_charges, save_energy, save_dos
from .chg import train_chg_model, load_pretrained_chg_model, infer_charges
from .energy import train_energy_model, load_pretrained_energy_model, infer_energy
from .dos import train_dos_model, load_pretrained_dos_model, infer_dos
from .utils import silence_deprecation_warnings


def parse_args():
    parser = argparse.ArgumentParser(
        prog="dftpy",
        description="PyTorch-based ML-DFT: charge, energy, DOS"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # “train” 子命令
    train_parser = subparsers.add_parser(
        "train",
        help="Train models: chg/energy/dos/all"
    )
    train_parser.add_argument(
        "--task", required=True, choices=["chg", "energy", "dos", "all"]
    )
    train_parser.add_argument(
        "--train-list", type=str, required=True,
        help="Train CSV，列名必须为 'files'"
    )
    train_parser.add_argument(
        "--val-list", type=str, required=True,
        help="Validation CSV，列名必须为 'files'"
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--grid-spacing", type=float, default=1.0)
    train_parser.add_argument("--cut-off-rad", type=float, default=6.0)
    train_parser.add_argument("--widest-gaussian", type=float, default=2.0)
    train_parser.add_argument("--narrowest-gaussian", type=float, default=0.5)
    train_parser.add_argument("--num-gamma", type=int, default=10)
    train_parser.add_argument(
        "--padding-multiplier",
        type=float,
        default=1.0,
        help="padding_size = ceil(max_atom_count * padding_multiplier)"
    )

    # “infer” 子命令
    infer_parser = subparsers.add_parser(
        "infer",
        help="Infer on new structures"
    )
    infer_parser.add_argument(
        "--infer-list", type=str, required=True,
        help="Predict CSV，列名必须为 'file_loc_test'"
    )
    infer_parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="预测结果输出目录"
    )
    infer_parser.add_argument("--predict-chg", action="store_true")
    infer_parser.add_argument("--predict-energy", action="store_true")
    infer_parser.add_argument("--predict-dos", action="store_true")
    infer_parser.add_argument("--grid-spacing", type=float, default=1.0)
    infer_parser.add_argument("--cut-off-rad", type=float, default=6.0)
    infer_parser.add_argument("--widest-gaussian", type=float, default=2.0)
    infer_parser.add_argument("--narrowest-gaussian", type=float, default=0.5)
    infer_parser.add_argument("--num-gamma", type=int, default=10)
    infer_parser.add_argument(
        "--padding-multiplier",
        type=float,
        default=1.0,
        help="padding_size = ceil(max_atom_count * padding_multiplier)"
    )
    infer_parser.add_argument(
        "--plot-dos",
        action="store_true",
        help="预测 DOS 结果后生成绘图（需要 matplotlib）"
    )

    return parser.parse_args()


def main():
    silence_deprecation_warnings()
    args = parse_args()

    if args.mode == "train":
        # 1) 读取 CSV，得到训练/验证文件夹列表
        train_folders = read_file_list(args.train_list, col="files")
        val_folders = read_file_list(args.val_list, col="files")

        # 2) 计算 padding_size：padding_multiplier * 最大原子数
        max_train = get_max_atom_count(train_folders)
        max_val = get_max_atom_count(val_folders)
        padding_size = int(max(max_train, max_val) * args.padding_multiplier)
        if padding_size < 1:
            raise ValueError("Computed padding_size < 1; 请检查输入文件或 padding_multiplier。")

        # 3) 按 task 依次训练
        if args.task in ("chg", "all"):
            print("[INFO] 训练电荷模型 …")
            train_chg_model(train_folders, val_folders, padding_size, args)

        if args.task in ("energy", "all"):
            print("[INFO] 训练能量模型 …")
            # 先加载电荷模型 checkpoint，若失败则提示
            chg_ckpt = "best_chg.pth"
            try:
                chg_model = load_pretrained_chg_model(chg_ckpt, padding_size)
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载电荷模型权重: {e}")
                return
            train_energy_model(train_folders, val_folders, chg_model, padding_size, args)

        if args.task in ("dos", "all"):
            print("[INFO] 训练 DOS 模型 …")
            chg_ckpt = "best_chg.pth"
            try:
                chg_model = load_pretrained_chg_model(chg_ckpt, padding_size)
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载电荷模型权重: {e}")
                return
            # 如果 DOS 训练依赖能量模型，则在此加载 energy_ckpt
            # 本示例假设不需要预训练 energy，当需要时请自行传入 checkpoint 路径
            dos_ckpt_pre = None
            train_dos_model(train_folders, val_folders, chg_model, dos_ckpt_pre, padding_size, args)

    elif args.mode == "infer":
        # 1) 读取 CSV，得到要预测的文件夹列表
        infer_folders = read_file_list(args.infer_list, col="file_loc_test")
        if not infer_folders:
            print("[WARN] infer-list 为空，没有任何文件夹可预测。")
            return

        os.makedirs(args.output_dir, exist_ok=True)

        # 2) 计算 padding_size 用于所有模型
        max_count = get_max_atom_count(infer_folders)
        padding_size = int(max_count * args.padding_multiplier)
        if padding_size < 1:
            raise ValueError("Computed padding_size < 1; 请检查输入文件或 padding_multiplier。")

        # 3) 先加载所需模型（如果相应 flag 被设置）
        chg_model = None
        energy_model = None
        dos_model = None

        if args.predict_chg:
            chg_ckpt = "best_chg.pth"
            try:
                chg_model = load_pretrained_chg_model(chg_ckpt, padding_size)
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载电荷模型权重: {e}")
                return

        if args.predict_energy:
            energy_ckpt = "best_energy.pth"
            try:
                energy_model = load_pretrained_energy_model(
                    energy_ckpt, padding_size
                )
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载能量模型权重: {e}")
                return

        if args.predict_dos:
            dos_ckpt = "best_dos.pth"
            try:
                dos_model = load_pretrained_dos_model(
                    dos_ckpt, padding_size
                )
            except FileNotFoundError as e:
                print(f"[ERROR] 无法加载 DOS 模型权重: {e}")
                return

        # 4) 循环对每个结构做预测并保存结果
        for folder in infer_folders:
            print(f"[INFO] 正在处理: {folder}")
            base = os.path.basename(folder.rstrip("/"))

            # —— 电荷预测 —— 
            if args.predict_chg:
                try:
                    chg_vals = infer_charges(
                        folder, chg_model, padding_size, args
                    )
                    save_charges(
                        chg_vals,
                        os.path.join(args.output_dir, f"charges_{base}.txt")
                    )
                except Exception as e:
                    warnings.warn(f"[WARN] 电荷预测失败 ({folder}): {e}")

            # —— 能量 & 力 & 应力预测 —— 
            if args.predict_energy:
                try:
                    e_val, forces, stress = infer_energy(
                        folder, chg_model, energy_model, padding_size, args
                    )
                    save_energy(
                        e_val,
                        forces,
                        stress,
                        os.path.join(args.output_dir, f"energy_{base}.txt")
                    )
                except Exception as e:
                    warnings.warn(f"[WARN] 能量预测失败 ({folder}): {e}")

            # —— DOS & 带隙预测 —— 
            if args.predict_dos:
                try:
                    energy_grid, dos_curve, vb, cb, bg, uncertainty = infer_dos(
                        folder, chg_model, dos_model, padding_size, args
                    )
                    save_dos(
                        energy_grid,
                        dos_curve,
                        vb,
                        cb,
                        bg,
                        os.path.join(args.output_dir, f"dos_{base}.txt")
                    )
                    if args.plot_dos:
                        # plot_dos 函数请自行在 utils 中实现，如需显示或保存图片
                        from .utils import plot_dos
                        plot_dos(
                            energy_grid,
                            dos_curve,
                            vb, cb,
                            os.path.join(args.output_dir, f"dos_plot_{base}.png")
                        )
                except Exception as e:
                    warnings.warn(f"[WARN] DOS 预测失败 ({folder}): {e}")

    else:
        raise ValueError("Unknown mode. 可用选项： 'train' 或 'infer'。")


if __name__ == "__main__":
    main()
