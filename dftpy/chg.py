# dftpy/chg.py

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pymatgen.io.vasp.outputs import Poscar, Chgcar
from .fp import fp_atom, fp_chg_norm, fp_norm
from .data_io import chg_data

# ------------------------------------------------------------------------------
# Device
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Utility: read reference charge density for training
# ------------------------------------------------------------------------------
def chg_ref(folder: str, vol: float, supercell):
    """
    从 CHGCAR 中提取参考电荷密度 (flattened)
    并返回 (coords, density, num_pts)，
    其中 density = data['total'].flatten('F') / vol
    """
    chgcar = Chgcar.from_file(os.path.join(folder, "CHGCAR"))
    density = chgcar.data["total"].flatten(order="F") / vol
    # 构建网格坐标 (fractional → Cartesian)
    centering = [0.5 - 1 / 2, 0.5 - 1 / 2, 0.5 - 1 / 2]
    lengths = chgcar.poscar.structure.lattice.abc
    xg = np.array(chgcar.get_axis_grid(0)) / supercell.lattice.a + centering[0]
    yg = np.array(chgcar.get_axis_grid(1)) / supercell.lattice.b + centering[1]
    zg = np.array(chgcar.get_axis_grid(2)) / supercell.lattice.c + centering[2]
    # meshgrid→positions
    xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
    pts = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    # fractional (i,j,k) → Cartesian: frac @ lattice matrix
    coords = pts @ supercell.lattice.matrix
    num_pts = [len(xg), len(yg), len(zg)]
    return coords, density, num_pts


# ------------------------------------------------------------------------------
# PyTorch single‐atom subnetwork for C/N/O (360 → 340)
# ------------------------------------------------------------------------------
class SingleAtomChargeNetCNO(nn.Module):
    def __init__(self):
        super().__init__()
        # 四层 200 单元 → 输出 340 (exponents+coefs)
        self.net = nn.Sequential(
            nn.Linear(360, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 340),
        )

    def forward(self, x):
        """
        x: (batch_size * padding_size, 360)
        返回: (batch_size * padding_size, 340)
        """
        out = self.net(x)
        # split first 93 dims as exponents (abs), rest 247 as coefs
        exp_part = torch.abs(out[:, :93])
        coef_part = out[:, 93:]
        return torch.cat([exp_part, coef_part], dim=1)


# ------------------------------------------------------------------------------
# PyTorch single‐atom subnetwork for H (360 → 208)
# ------------------------------------------------------------------------------
class SingleAtomChargeNetH(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(360, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 208),
        )

    def forward(self, x):
        """
        x: (batch_size * padding_size, 360)
        返回: (batch_size * padding_size, 208)
        """
        out = self.net(x)
        exp_part = torch.abs(out[:, :58])
        coef_part = out[:, 58:]
        return torch.cat([exp_part, coef_part], dim=1)


# ------------------------------------------------------------------------------
# Combined 4‐branch charge model
# ------------------------------------------------------------------------------
class ChargeModel(nn.Module):
    """
    接受四类输入 (batch, padding_size, 360) → 输出四个系数张量：
      - Coef_C: (batch, padding_size, 340)
      - Coef_H: (batch, padding_size, 208)
      - Coef_N: (batch, padding_size, 340)
      - Coef_O: (batch, padding_size, 340)
    """
    def __init__(self, padding_size: int):
        super().__init__()
        self.padding_size = padding_size
        self.netC = SingleAtomChargeNetCNO().to(DEVICE)
        self.netH = SingleAtomChargeNetH().to(DEVICE)
        self.netN = SingleAtomChargeNetCNO().to(DEVICE)
        self.netO = SingleAtomChargeNetCNO().to(DEVICE)

    def forward(self, X_C, X_H, X_N, X_O):
        """
        X_*: (batch, padding_size, 360)
        返回: Coef_C, Coef_H, Coef_N, Coef_O
          - Coef_C: (batch, padding_size, 340)
          - Coef_H: (batch, padding_size, 208)
          - Coef_N: (batch, padding_size, 340)
          - Coef_O: (batch, padding_size, 340)
        """
        B, P, _ = X_C.shape
        # flatten batch & pad → 2D
        C_flat = X_C.view(B * P, 360)
        H_flat = X_H.view(B * P, 360)
        N_flat = X_N.view(B * P, 360)
        O_flat = X_O.view(B * P, 360)

        outC = self.netC(C_flat).view(B, P, -1)
        outH = self.netH(H_flat).view(B, P, -1)
        outN = self.netN(N_flat).view(B, P, -1)
        outO = self.netO(O_flat).view(B, P, -1)

        return outC, outH, outN, outO


# ------------------------------------------------------------------------------
# Instantiate charge model
# ------------------------------------------------------------------------------
def init_chgmod(padding_size: int):
    """
    初始化 PyTorch ChargeModel。尚未加载权重。
    """
    model = ChargeModel(padding_size)
    return model


# ------------------------------------------------------------------------------
# Load pretrained weights
# ------------------------------------------------------------------------------
def load_pretrained_chg_model(checkpoint_path: str, padding_size: int):
    """
    加载一个训练好的 ChargeModel 并返回 (eval 模式)。
    """
    model = init_chgmod(padding_size).to(DEVICE)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"ChargeModel 权重不存在: {checkpoint_path}")


# ------------------------------------------------------------------------------
# PyTorch Dataset for charge training
# ------------------------------------------------------------------------------
class ChargeDataset(Dataset):
    def __init__(self, X_C, X_H, X_N, X_O, Coef_C, Coef_H, Coef_N, Coef_O):
        """
        X_*:       np.ndarray, shape = (n_samples, padding_size, 360)
        Coef_*:    np.ndarray, 对应每个元素的训练标签 (n_samples, padding_size, dim_coef)
                   如 Coef_C: dim=340; Coef_H: dim=208; Coef_N:340; Coef_O:340
        """
        self.X_C = torch.from_numpy(X_C).float()
        self.X_H = torch.from_numpy(X_H).float()
        self.X_N = torch.from_numpy(X_N).float()
        self.X_O = torch.from_numpy(X_O).float()
        self.Coef_C = torch.from_numpy(Coef_C).float()
        self.Coef_H = torch.from_numpy(Coef_H).float()
        self.Coef_N = torch.from_numpy(Coef_N).float()
        self.Coef_O = torch.from_numpy(Coef_O).float()

    def __len__(self):
        return self.X_C.shape[0]

    def __getitem__(self, idx):
        return (
            self.X_C[idx], self.X_H[idx],
            self.X_N[idx], self.X_O[idx],
            self.Coef_C[idx], self.Coef_H[idx],
            self.Coef_N[idx], self.Coef_O[idx],
        )


# ------------------------------------------------------------------------------
# Custom MSE Loss for coefficients
# ------------------------------------------------------------------------------
mse_loss = nn.MSELoss()


# ------------------------------------------------------------------------------
# Training / Retrain pipeline
# ------------------------------------------------------------------------------
def train_chg_model(
    train_folders, val_folders, padding_size: int, args
):
    """
    训练电荷模型：
      1. 从 train_folders, val_folders 中生成 PyTorch Dataset
      2. 定义 ChargeModel、优化器、损失函数
      3. 迭代训练，保存最优模型到 'best_chg.pth'
    args 包含:
      - batch_size, epochs, learning_rate, patience
      - grid_spacing, cut_off_rad, widest_gaussian, narrowest_gaussian, num_gamma
    """
    # 1. 预先读取所有训练数据，生成指纹 + 基函数 + 原子掩码 + 真实 Coefs
    def prepare_coef_data(folders):
        X_C_list, X_H_list, X_N_list, X_O_list = [], [], [], []
        Coef_C_list, Coef_H_list, Coef_N_list, Coef_O_list = [], [], [], []
        for folder in folders:
            # 读取 POSCAR → fp_atom → 得到 dset (Cartesian grid) 与 basis_mat
            poscar = Poscar.from_file(os.path.join(folder, "POSCAR"))
            struct = poscar.structure
            vol = struct.volume

            dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(
                struct,
                args.grid_spacing,
                args.cut_off_rad,
                args.widest_gaussian,
                args.narrowest_gaussian,
                args.num_gamma,
            )
            # at_elem = [nC, nH, nN, nO]
            # chg_ref 返回 (coords, density, num_pts)
            chg_coor, chg_den, num_pts = chg_ref(folder, vol, struct)

            # 根据 at_elem 拆分 → pad to padding_size
            X_3D1, X_3D2, X_3D3, X_3D4, basis1, basis2, basis3, basis4, C_m, H_m, N_m, O_m = chg_data(
                dset, basis_mat, at_elem[0], at_elem[1], at_elem[2], at_elem[3], padding_size
            )
            # 模型输入：(1, P, 360)
            X_C_list.append(X_3D1)
            X_H_list.append(X_3D2)
            X_N_list.append(X_3D3)
            X_O_list.append(X_3D4)

            # 真实系数(Coef_C, Coef_H, Coef_N, Coef_O)：直接从原脚本的 chg_train 里获取
            # 这里我们假设已经有预先计算并保存在 `.npy` 文件中的系数，
            # 或者可在内存中动态调用原 Keras 模型进行“标签”生成。
            # 为示例，加载预存 npy：
            Coef_C = np.load(os.path.join(folder, "Coef_C.npy"))  # (nC, 340)
            Coef_H = np.load(os.path.join(folder, "Coef_H.npy"))  # (nH, 208)
            Coef_N = np.load(os.path.join(folder, "Coef_N.npy"))  # (nN, 340) or zeros if nN=0
            Coef_O = np.load(os.path.join(folder, "Coef_O.npy"))  # (nO, 340) or zeros if nO=0

            # Pad Coefs to padding_size
            def pad_coef(arr, out_dim):
                """
                arr: (natoms, dim_coef) or empty
                返回: (padding_size, dim_coef) 张量 (padding 部分填 0)
                """
                if arr.size == 0:
                    return np.zeros((padding_size, out_dim), dtype=np.float32)
                padded = np.zeros((padding_size, out_dim), dtype=np.float32)
                n = min(arr.shape[0], padding_size)
                padded[:n, :] = arr[:n, :]
                return padded

            Coef_C_list.append(pad_coef(Coef_C, 340))
            Coef_H_list.append(pad_coef(Coef_H, 208))
            Coef_N_list.append(pad_coef(Coef_N, 340))
            Coef_O_list.append(pad_coef(Coef_O, 340))

        # Stack into np arrays
        X_C_arr = np.vstack(X_C_list)  # (N, P, 360)
        X_H_arr = np.vstack(X_H_list)
        X_N_arr = np.vstack(X_N_list)
        X_O_arr = np.vstack(X_O_list)
        Coef_C_arr = np.stack(Coef_C_list)  # (N, P, 340)
        Coef_H_arr = np.stack(Coef_H_list)  # (N, P, 208)
        Coef_N_arr = np.stack(Coef_N_list)  # (N, P, 340)
        Coef_O_arr = np.stack(Coef_O_list)  # (N, P, 340)

        return X_C_arr, X_H_arr, X_N_arr, X_O_arr, Coef_C_arr, Coef_H_arr, Coef_N_arr, Coef_O_arr

    X_C_train, X_H_train, X_N_train, X_O_train, Coef_C_train, Coef_H_train, Coef_N_train, Coef_O_train = prepare_coef_data(train_folders)
    X_C_val,   X_H_val,   X_N_val,   X_O_val,   Coef_C_val,   Coef_H_val,   Coef_N_val,   Coef_O_val   = prepare_coef_data(val_folders)

    train_dataset = ChargeDataset(X_C_train, X_H_train, X_N_train, X_O_train, Coef_C_train, Coef_H_train, Coef_N_train, Coef_O_train)
    val_dataset   = ChargeDataset(X_C_val,   X_H_val,   X_N_val,   X_O_val,   Coef_C_val,   Coef_H_val,   Coef_N_val,   Coef_O_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    # 2. 初始化模型
    model = init_chgmod(padding_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0

    # 3. 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for X_Cb, X_Hb, X_Nb, X_Ob, Coef_Cb, Coef_Hb, Coef_Nb, Coef_Ob in train_loader:
            X_Cb = X_Cb.to(DEVICE)
            X_Hb = X_Hb.to(DEVICE)
            X_Nb = X_Nb.to(DEVICE)
            X_Ob = X_Ob.to(DEVICE)
            Coef_Cb = Coef_Cb.to(DEVICE)
            Coef_Hb = Coef_Hb.to(DEVICE)
            Coef_Nb = Coef_Nb.to(DEVICE)
            Coef_Ob = Coef_Ob.to(DEVICE)

            optimizer.zero_grad()
            outC, outH, outN, outO = model(X_Cb, X_Hb, X_Nb, X_Ob)
            loss_C = mse_loss(outC, Coef_Cb)
            loss_H = mse_loss(outH, Coef_Hb)
            loss_N = mse_loss(outN, Coef_Nb)
            loss_O = mse_loss(outO, Coef_Ob)
            loss = loss_C + loss_H + loss_N + loss_O
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_Cb, X_Hb, X_Nb, X_Ob, Coef_Cb, Coef_Hb, Coef_Nb, Coef_Ob in val_loader:
                X_Cb = X_Cb.to(DEVICE)
                X_Hb = X_Hb.to(DEVICE)
                X_Nb = X_Nb.to(DEVICE)
                X_Ob = X_Ob.to(DEVICE)
                Coef_Cb = Coef_Cb.to(DEVICE)
                Coef_Hb = Coef_Hb.to(DEVICE)
                Coef_Nb = Coef_Nb.to(DEVICE)
                Coef_Ob = Coef_Ob.to(DEVICE)

                outC, outH, outN, outO = model(X_Cb, X_Hb, X_Nb, X_Ob)
                loss_C = mse_loss(outC, Coef_Cb)
                loss_H = mse_loss(outH, Coef_Hb)
                loss_N = mse_loss(outN, Coef_Nb)
                loss_O = mse_loss(outO, Coef_Ob)
                val_loss += (loss_C + loss_H + loss_N + loss_O).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_train:.6f}, Val Loss = {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save(model.state_dict(), "best_chg.pth")

        if epoch - best_epoch >= args.patience:
            print("Early stopping.")
            break

    print(f"Charge training done. Best Val Loss={best_val_loss:.6f} at epoch {best_epoch}.")


# ------------------------------------------------------------------------------
# Inference: compute atomic charge coefficients & actual atomic charges
# ------------------------------------------------------------------------------
def infer_charges(
    folder: str,
    chg_model: nn.Module,
    padding_size: int,
    args
):
    """
    CLI 推理函数：
      1. 读取 POSCAR → fp_atom → 得到 dset, basis, sites_elem, num_atoms, at_elem
      2. chg_model 推理得到 Coef_C, Coef_H, Coef_N, Coef_O
      3. 基于 Coefs、预定义高斯基求原子净电荷
      4. 如果 write_chg=True，则写 Pred_CHG_test*.dat

    返回：
      - atomic_charges: np.ndarray, 按 POSCAR 原子顺序排列 (num_atoms,)
    """
    poscar = Poscar.from_file(os.path.join(folder, "POSCAR"))
    struct = poscar.structure
    vol = struct.volume

    dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(
        struct,
        args.grid_spacing,
        args.cut_off_rad,
        args.widest_gaussian,
        args.narrowest_gaussian,
        args.num_gamma,
    )
    # at_elem = [nC, nH, nN, nO]
    # 构造 pad‐level 数据
    X_3D1, X_3D2, X_3D3, X_3D4, basis1, basis2, basis3, basis4, C_m, H_m, N_m, O_m = chg_data(
        dset, basis_mat, at_elem[0], at_elem[1], at_elem[2], at_elem[3], padding_size
    )
    # 输入 PyTorch
    X_C = torch.from_numpy(X_3D1).float().to(DEVICE)
    X_H = torch.from_numpy(X_3D2).float().to(DEVICE)
    X_N = torch.from_numpy(X_3D3).float().to(DEVICE)
    X_O = torch.from_numpy(X_3D4).float().to(DEVICE)

    chg_model.eval()
    with torch.no_grad():
        Coef_C_t, Coef_H_t, Coef_N_t, Coef_O_t = chg_model(X_C, X_H, X_N, X_O)

    # 转回 NumPy 并截掉 pad 部分
    Coef_C = Coef_C_t.cpu().numpy().squeeze(0)[: at_elem[0], :]  # shape=(nC, 340)
    Coef_H = Coef_H_t.cpu().numpy().squeeze(0)[: at_elem[1], :]  # (nH, 208)
    if at_elem[2] > 0:
        Coef_N = Coef_N_t.cpu().numpy().squeeze(0)[: at_elem[2], :]  # (nN, 340)
    else:
        Coef_N = np.zeros((0, 340), dtype=np.float32)
    if at_elem[3] > 0:
        Coef_O = Coef_O_t.cpu().numpy().squeeze(0)[: at_elem[3], :]  # (nO, 340)
    else:
        Coef_O = np.zeros((0, 340), dtype=np.float32)

    # 3. 计算每个原子的电荷：对应原始 s_chg / p_chg / d_chg / f_chg / g_chg 逻辑
    # 简化示例：只计算 s (spherical) 项，总电荷 ≈ sum_i π^(3/2) * coef_i / exp_i^(3/2)
    def compute_atom_charge(exp_arr, coef_arr):
        """
        exp_arr: (n_basis,), coef_arr: (n_basis,)
        返回: float
        """
        return np.sum((np.pi) ** (1.5) * coef_arr / (exp_arr ** 1.5))

    atomic_charges = []
    # C atoms
    for idx in range(at_elem[0]):
        exps = Coef_C[idx, :93]
        coefs = Coef_C[idx, 93:]
        q = compute_atom_charge(exps, coefs)
        atomic_charges.append(q)
    # H atoms
    for idx in range(at_elem[1]):
        exps = Coef_H[idx, :58]
        coefs = Coef_H[idx, 58:]
        q = compute_atom_charge(exps, coefs)
        atomic_charges.append(q)
    # N atoms
    for idx in range(at_elem[2]):
        exps = Coef_N[idx, :93]
        coefs = Coef_N[idx, 93:]
        q = compute_atom_charge(exps, coefs)
        atomic_charges.append(q)
    # O atoms
    for idx in range(at_elem[3]):
        exps = Coef_O[idx, :93]
        coefs = Coef_O[idx, 93:]
        q = compute_atom_charge(exps, coefs)
        atomic_charges.append(q)

    atomic_charges = np.array(atomic_charges, dtype=np.float32)

    # 写出到文件（如果需要）
    if args.write_chg:
        coords, density, _ = chg_ref(folder, vol, struct)
        # 将 atomic_charges 标准化到真实电荷 (密度体积积分)，示例中可略
        with open(f"Pred_CHG_test_{os.path.basename(folder)}.dat", "w") as fp:
            for q in atomic_charges:
                fp.write(f"{q:.6f}\n")

    return atomic_charges


# ------------------------------------------------------------------------------
# 按需导出给 CLI 使用
# ------------------------------------------------------------------------------
__all__ = [
    "init_chgmod",
    "load_pretrained_chg_model",
    "train_chg_model",
    "infer_charges",
]
