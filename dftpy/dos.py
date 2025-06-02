# dftpy/dos.py

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pymatgen.io.vasp.outputs import Poscar
from .fp import fp_chg_norm, fp_norm
from .data_io import dos_mask

# ------------------------------------------------------------------------------
# Device
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Utility: read DOS reference and VB/CB for training
# ------------------------------------------------------------------------------
def dos_data(folder: str, total_elec: float):
    """
    读取训练标签：
      - Prop_dos: np.ndarray, shape=(343,), 对应 dos 文件中 343 个能量点的密度
      - VB: float, 价带顶 (从 VB_CB 文件)
      - CB: float, 导带底
    """
    dos_file = os.path.join(folder, "dos")
    dos_vals = [float(line.split()[0]) for line in open(dos_file).readlines()]
    Prop = np.array(dos_vals, dtype=np.float32) / total_elec

    vbcb_file = os.path.join(folder, "VB_CB")
    lines = [list(map(float, l.split())) for l in open(vbcb_file).readlines()]
    VB, CB = abs(lines[0][0]), abs(lines[1][0])
    return Prop, VB, CB


# ------------------------------------------------------------------------------
# Single‐atom DOS subnetworks
# ------------------------------------------------------------------------------
class SingleAtomDOSCNO(nn.Module):
    """
    针对 C/N/O 原子：
    输入维度 = 360 (fprint) + 340 (charge‐aug) = 700
    输出维度 = 343 (DOS 在各能级的贡献)
    """
    def __init__(self, input_dim=700, hidden_dim=600, out_dim=343):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )
        # 1D 卷积层：kernel_size=3, in_channels=1, out_channels=3，但 PyTorch 要调整格式
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: (batch * padding_size, 700)
        返回: (batch * padding_size, 343)
        """
        h = self.net(x)                      # → (B*P, 343)
        h = h.view(-1, 1, 343)               # → (B*P, 1, 343)
        h = self.conv1d(h)                   # → (B*P, 3, 343)
        h = h.mean(dim=1)                    # → (B*P, 343)  沿着 conv_out_channels 取平均
        return h


class SingleAtomDOSH(nn.Module):
    """
    针对 H 原子：
    输入维度 = 360 (fprint) + 208 (charge‐aug) = 568
    输出维度 = 343
    """
    def __init__(self, input_dim=568, hidden_dim=600, out_dim=343):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: (batch * padding_size, 568)
        返回: (batch * padding_size, 343)
        """
        h = self.net(x)
        h = h.view(-1, 1, 343)
        h = self.conv1d(h)
        h = h.mean(dim=1)
        return h


# ------------------------------------------------------------------------------
# Combined DOS model
# ------------------------------------------------------------------------------
class DOSModel(nn.Module):
    """
    接受:
      X_C  (batch, P, 700)
      X_H  (batch, P, 568)
      X_N  (batch, P, 700)
      X_O  (batch, P, 700)
      total_elec (batch, 1)
      C_d, H_d, N_d, O_d: mask tensors, size (batch, P, 343)
    输出:
      - Pred_dos: (batch, 343)    # 归一化 DOS 曲线
      - VB_CB: (batch, 2)         # [VB, CB]
    """
    def __init__(self, padding_size: int):
        super().__init__()
        self.padding_size = padding_size

        self.netC = SingleAtomDOSCNO(input_dim=360 + 340)
        self.netH = SingleAtomDOSH(input_dim=360 + 208)
        self.netN = SingleAtomDOSCNO(input_dim=360 + 340)
        self.netO = SingleAtomDOSCNO(input_dim=360 + 340)

        # 后续 MLP 用于预测 [VB, CB]
        self.mlp = nn.Sequential(
            nn.Linear(343, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, X_C, X_H, X_N, X_O, total_elec, C_d, H_d, N_d, O_d):
        """
        X_*:       (batch, P, feat_dim)
        total_elec: (batch, 1)
        C_d etc:   (batch, P, 343)
        """
        B, P, _ = X_C.shape

        # 展平每个分支到 (B*P, feat_dim)
        C_flat = X_C.reshape(B * P, -1)
        H_flat = X_H.reshape(B * P, -1)
        N_flat = X_N.reshape(B * P, -1)
        O_flat = X_O.reshape(B * P, -1)

        # 单原子前向
        outC = self.netC(C_flat).view(B, P, -1)  # → (B, P, 343)
        outH = self.netH(H_flat).view(B, P, -1)
        outN = self.netN(N_flat).view(B, P, -1)
        outO = self.netO(O_flat).view(B, P, -1)

        # 按 mask 乘以权重
        # C_d 等为 (B, P, 343)，按元素相乘
        weightedC = outC * C_d
        weightedH = outH * H_d
        weightedN = outN * N_d
        weightedO = outO * O_d

        # sum over atoms (dim=1)，得到 (B, 343)
        sumC = weightedC.sum(dim=1)
        sumH = weightedH.sum(dim=1)
        sumN = weightedN.sum(dim=1)
        sumO = weightedO.sum(dim=1)

        # 叠加所有元素
        raw_dos = sumC + sumH + sumN + sumO  # (B, 343)

        # 除以 total_elec 得到归一化 DOS
        norm_dos = raw_dos / total_elec

        # VB/CB 预测
        vbcb = self.mlp(norm_dos)  # (B, 2)

        return norm_dos, vbcb


# ------------------------------------------------------------------------------
# Initialize DOS model (未加载权重)
# ------------------------------------------------------------------------------
def init_DOSmod(padding_size: int):
    """
    初始化 DOSModel，不加载预训练权重
    """
    model = DOSModel(padding_size).to(DEVICE)
    return model


# ------------------------------------------------------------------------------
# Load pretrained DOS model
# ------------------------------------------------------------------------------
def load_pretrained_dos_model(checkpoint_path: str, padding_size: int):
    """
    加载已训练好的 DOSModel
    """
    model = init_DOSmod(padding_size)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"DOSModel 权重不存在: {checkpoint_path}")


# ------------------------------------------------------------------------------
# Dataset for DOS training
# ------------------------------------------------------------------------------
class DOSDataset(Dataset):
    def __init__(
        self,
        X_C, X_H, X_N, X_O,
        total_elec,
        C_d, H_d, N_d, O_d,
        Prop_dos, VB_CB,
    ):
        """
        参数：
          - X_*:       np.ndarray, shape=(N, P, feat_dim_per_atom)
          - total_elec: np.ndarray, shape=(N, 1)
          - C_d, H_d, N_d, O_d: np.ndarray, shape=(N, P, 343)
          - Prop_dos:     np.ndarray, shape=(N, 343)
          - VB_CB:        np.ndarray, shape=(N, 2)
        """
        self.X_C = torch.from_numpy(X_C).float()
        self.X_H = torch.from_numpy(X_H).float()
        self.X_N = torch.from_numpy(X_N).float()
        self.X_O = torch.from_numpy(X_O).float()
        self.total_elec = torch.from_numpy(total_elec).float()
        self.C_d = torch.from_numpy(C_d).float()
        self.H_d = torch.from_numpy(H_d).float()
        self.N_d = torch.from_numpy(N_d).float()
        self.O_d = torch.from_numpy(O_d).float()
        self.Prop_dos = torch.from_numpy(Prop_dos).float()
        self.VB_CB = torch.from_numpy(VB_CB).float()

    def __len__(self):
        return self.X_C.shape[0]

    def __getitem__(self, idx):
        return (
            self.X_C[idx],
            self.X_H[idx],
            self.X_N[idx],
            self.X_O[idx],
            self.total_elec[idx],
            self.C_d[idx],
            self.H_d[idx],
            self.N_d[idx],
            self.O_d[idx],
            self.Prop_dos[idx],
            self.VB_CB[idx],
        )


# ------------------------------------------------------------------------------
# Loss: MSE for DOS (weighted heavily) + MSE for VB/CB
# ------------------------------------------------------------------------------
class DOSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_dos, true_dos, pred_vbcb, true_vbcb):
        """
        pred_dos, true_dos: (B, 343)
        pred_vbcb, true_vbcb: (B, 2)
        """
        loss_dos = self.mse(pred_dos, true_dos)
        loss_vbcb = self.mse(pred_vbcb, true_vbcb)
        # 给 DOS 损失较大权重 (如 1000:1)
        return 1000 * loss_dos + loss_vbcb


# ------------------------------------------------------------------------------
# Training pipeline for DOS
# ------------------------------------------------------------------------------
def train_dos_model(
    train_folders, val_folders, padding_size: int, args
):
    """
    训练 DOS 模型：
      1. 从 train_folders, val_folders 中读取数据，计算 fp_chg_norm + dos_mask
      2. 构造 Dataset, DataLoader
      3. 初始化 DOSModel, 定义优化器和 DOSLoss
      4. 迭代训练，保存最优权重到 'best_dos.pth'
    args 包含: batch_size, epochs, learning_rate, patience, grid_spacing, cut_off_rad, ...)
    """
    def prepare_dos_data(folders):
        X_C_list, X_H_list, X_N_list, X_O_list = [], [], [], []
        total_e_list = []
        C_d_list, H_d_list, N_d_list, O_d_list = [], [], [], []
        Prop_dos_list, VB_CB_list = [], []

        for folder in folders:
            # 读取 POSCAR → fp_atom → 得到 dset 和 basis
            poscar = Poscar.from_file(os.path.join(folder, "POSCAR"))
            struct = poscar.structure
            vol = struct.volume

            # 1. 先预测电荷系数（可复用载入的 ChargeModel），然后生成 charge‐aug features
            #    这里假设电荷预测已经完成，并存为.npy
            Coef_C = np.load(os.path.join(folder, "Coef_C.npy"))  # (nC, 340)
            Coef_H = np.load(os.path.join(folder, "Coef_H.npy"))  # (nH, 208)
            Coef_N = np.load(os.path.join(folder, "Coef_N.npy"))  # (nN, 340)
            Coef_O = np.load(os.path.join(folder, "Coef_O.npy"))  # (nO, 340)

            dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(
                struct,
                args.grid_spacing,
                args.cut_off_rad,
                args.widest_gaussian,
                args.narrowest_gaussian,
                args.num_gamma,
            )

            # 生成电荷归一化的输入特征
            X_C3, X_H3, X_N3, X_O3 = fp_chg_norm(
                Coef_C[np.newaxis, :, :],   # 变成 (1, nC, 340)
                Coef_H[np.newaxis, :, :],   # (1, nH, 208)
                Coef_N[np.newaxis, :, :],   # ...
                Coef_O[np.newaxis, :, :],
                dset[np.newaxis, :, :],     # (1, num_atoms, 360)
                at_elem[0], at_elem[1], at_elem[2], at_elem[3],
                padding_size,
            )
            # fp_chg_norm 输出：每个 atom 类型的特征 shape=(1, P, feat_dim)
            # 直接提取并转为 numpy
            X_C_arr = X_C3.squeeze(0)  # (P, 700)
            X_H_arr = X_H3.squeeze(0)  # (P, 568)
            X_N_arr = X_N3.squeeze(0)  # (P, 700)
            X_O_arr = X_O3.squeeze(0)  # (P, 700)

            # 创建 mask：dos_mask 返回 (1, P, 341) for each type
            C_d_mask, H_d_mask, N_d_mask, O_d_mask = dos_mask(
                X_C_arr[np.newaxis, :, :],  # dummy for shape (1, P, 700)
                X_H_arr[np.newaxis, :, :],
                X_N_arr[np.newaxis, :, :],
                X_O_arr[np.newaxis, :, :],
                padding_size,
            )
            C_d_list.append(C_d_mask.squeeze(0))
            H_d_list.append(H_d_mask.squeeze(0))
            N_d_list.append(N_d_mask.squeeze(0))
            O_d_list.append(O_d_mask.squeeze(0))

            X_C_list.append(X_C_arr)
            X_H_list.append(X_H_arr)
            X_N_list.append(X_N_arr)
            X_O_list.append(X_O_arr)

            # total_elec
            electrons = [ {6:4, 1:1, 7:5, 8:6}[z] for z in struct.atomic_numbers ]
            total_e = np.sum(electrons, dtype=np.float32)
            total_e_list.append(np.array([total_e], dtype=np.float32))

            # 标签: Prop_dos (343)，VB/CB (2)
            Prop_dos, VB, CB = dos_data(folder, total_e)
            Prop_dos_list.append(Prop_dos)
            VB_CB_list.append(np.array([ -VB, -CB ], dtype=np.float32))  
            # 注意：原 Keras 用 -1*VB, -1*CB

        # 转为 numpy arrays
        X_C_arr = np.stack(X_C_list)        # (N, P, 700)
        X_H_arr = np.stack(X_H_list)        # (N, P, 568)
        X_N_arr = np.stack(X_N_list)        # (N, P, 700)
        X_O_arr = np.stack(X_O_list)        # (N, P, 700)
        total_e_arr = np.stack(total_e_list)  # (N, 1)
        C_d_arr = np.stack(C_d_list)          # (N, P, 341)
        H_d_arr = np.stack(H_d_list)
        N_d_arr = np.stack(N_d_list)
        O_d_arr = np.stack(O_d_list)
        Prop_dos_arr = np.stack(Prop_dos_list)  # (N, 343)
        VB_CB_arr = np.stack(VB_CB_list)        # (N, 2)

        return (
            X_C_arr, X_H_arr, X_N_arr, X_O_arr,
            total_e_arr,
            C_d_arr, H_d_arr, N_d_arr, O_d_arr,
            Prop_dos_arr, VB_CB_arr,
        )

    # 1. 准备训练/验证集
    data_train = prepare_dos_data(train_folders)
    data_val   = prepare_dos_data(val_folders)

    train_dataset = DOSDataset(*data_train)
    val_dataset   = DOSDataset(*data_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    # 2. 初始化模型 & 损失 & 优化器
    model = init_DOSmod(padding_size).to(DEVICE)
    criterion = DOSLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0

    # 3. 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for (
            X_Cb, X_Hb, X_Nb, X_Ob,
            total_e_b,
            C_db, H_db, N_db, O_db,
            Prop_b, VB_CB_b,
        ) in train_loader:
            X_Cb = X_Cb.to(DEVICE)
            X_Hb = X_Hb.to(DEVICE)
            X_Nb = X_Nb.to(DEVICE)
            X_Ob = X_Ob.to(DEVICE)
            total_e_b = total_e_b.to(DEVICE)
            C_db = C_db.to(DEVICE)
            H_db = H_db.to(DEVICE)
            N_db = N_db.to(DEVICE)
            O_db = O_db.to(DEVICE)
            Prop_b = Prop_b.to(DEVICE)
            VB_CB_b = VB_CB_b.to(DEVICE)

            optimizer.zero_grad()
            pred_dos, pred_vbcb = model(
                X_Cb, X_Hb, X_Nb, X_Ob,
                total_e_b, C_db, H_db, N_db, O_db,
            )
            loss = criterion(pred_dos, Prop_b, pred_vbcb, VB_CB_b)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for (
                X_Cb, X_Hb, X_Nb, X_Ob,
                total_e_b,
                C_db, H_db, N_db, O_db,
                Prop_b, VB_CB_b,
            ) in val_loader:
                X_Cb = X_Cb.to(DEVICE)
                X_Hb = X_Hb.to(DEVICE)
                X_Nb = X_Nb.to(DEVICE)
                X_Ob = X_Ob.to(DEVICE)
                total_e_b = total_e_b.to(DEVICE)
                C_db = C_db.to(DEVICE)
                H_db = H_db.to(DEVICE)
                N_db = N_db.to(DEVICE)
                O_db = O_db.to(DEVICE)
                Prop_b = Prop_b.to(DEVICE)
                VB_CB_b = VB_CB_b.to(DEVICE)

                pred_dos, pred_vbcb = model(
                    X_Cb, X_Hb, X_Nb, X_Ob,
                    total_e_b, C_db, H_db, N_db, O_db,
                )
                val_loss = criterion(pred_dos, Prop_b, pred_vbcb, VB_CB_b)
                total_val_loss += val_loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_train:.6f}, Val Loss = {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save(model.state_dict(), "best_dos.pth")

        if epoch - best_epoch >= args.patience:
            print("Early stopping.")
            break

    print(f"DOS training done. Best Val Loss={best_val_loss:.6f} at epoch {best_epoch}.")


# ------------------------------------------------------------------------------
# Inference: predict DOS & VB/CB for新结构
# ------------------------------------------------------------------------------
def infer_dos(
    folder: str,
    chg_model: nn.Module,
    dos_model: nn.Module,
    padding_size: int,
    args
):
    """
    CLI 推理：
      1. 读取 POSCAR → fp_atom → 生成指纹 dset / basis
      2. 用 chg_model 推理 Coefs → 生成 X_C, X_H, X_N, X_O
      3. 构造 dos_mask
      4. dos_model 前向 → norm_dos, vbcb
      5. 返回 energy_grid, dos_curve, vb, cb, bandgap, uncertainty (暂不做 MC 重采样，只输出一次)
    """
    poscar = Poscar.from_file(os.path.join(folder, "POSCAR"))
    struct = poscar.structure
    vol = struct.volume

    # 1. 生成指纹 dset + basis
    dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(
        struct,
        args.grid_spacing,
        args.cut_off_rad,
        args.widest_gaussian,
        args.narrowest_gaussian,
        args.num_gamma,
    )

    # 2. 用 chg_model 推理电荷系数
    #    省略细节：假设 chg_model 已加载
    #    新版 chg_model.forward 接受四个 (1, P, 360) 输入 → 输出 Coef_C, Coef_H, Coef_N, Coef_O
    X_3D1, X_3D2, X_3D3, X_3D4, basis1, basis2, basis3, basis4, C_m, H_m, N_m, O_m = chg_data(
        dset, basis_mat, at_elem[0], at_elem[1], at_elem[2], at_elem[3], padding_size
    )
    # 转 PyTorch
    X_C_t = torch.from_numpy(X_3D1).float().to(DEVICE)
    X_H_t = torch.from_numpy(X_3D2).float().to(DEVICE)
    X_N_t = torch.from_numpy(X_3D3).float().to(DEVICE)
    X_O_t = torch.from_numpy(X_3D4).float().to(DEVICE)

    chg_model.eval()
    with torch.no_grad():
        Coef_C_t, Coef_H_t, Coef_N_t, Coef_O_t = chg_model(X_C_t, X_H_t, X_N_t, X_O_t)

    Coef_C = Coef_C_t.cpu().numpy().squeeze(0)[: at_elem[0], :]  # (nC, 340)
    Coef_H = Coef_H_t.cpu().numpy().squeeze(0)[: at_elem[1], :]  # (nH, 208)
    Coef_N = Coef_N_t.cpu().numpy().squeeze(0)[: at_elem[2], :] if at_elem[2] > 0 else np.zeros((0, 340), dtype=np.float32)
    Coef_O = Coef_O_t.cpu().numpy().squeeze(0)[: at_elem[3], :] if at_elem[3] > 0 else np.zeros((0, 340), dtype=np.float32)

    # 3. 生成电荷归一化特征 (同训练时)
    X_C3, X_H3, X_N3, X_O3 = fp_chg_norm(
        Coef_C[np.newaxis, :, :],
        Coef_H[np.newaxis, :, :],
        Coef_N[np.newaxis, :, :],
        Coef_O[np.newaxis, :, :],
        dset[np.newaxis, :, :],
        at_elem[0], at_elem[1], at_elem[2], at_elem[3],
        padding_size,
    )
    X_C_arr = X_C3.squeeze(0)  # (P, 700)
    X_H_arr = X_H3.squeeze(0)  # (P, 568)
    X_N_arr = X_N3.squeeze(0)  # (P, 700)
    X_O_arr = X_O3.squeeze(0)  # (P, 700)

    # 4. 构建 dos_mask → (1, P, 341)
    C_d_mask, H_d_mask, N_d_mask, O_d_mask = dos_mask(
        X_C_arr[np.newaxis, :, :],
        X_H_arr[np.newaxis, :, :],
        X_N_arr[np.newaxis, :, :],
        X_O_arr[np.newaxis, :, :],
        padding_size,
    )

    # PyTorch tensor
    X_C_t2 = torch.from_numpy(X_C_arr[np.newaxis, :, :]).float().to(DEVICE)
    X_H_t2 = torch.from_numpy(X_H_arr[np.newaxis, :, :]).float().to(DEVICE)
    X_N_t2 = torch.from_numpy(X_N_arr[np.newaxis, :, :]).float().to(DEVICE)
    X_O_t2 = torch.from_numpy(X_O_arr[np.newaxis, :, :]).float().to(DEVICE)
    total_e = np.array([ np.sum([ {6:4,1:1,7:5,8:6}[z] for z in struct.atomic_numbers ], dtype=np.float32 ) ])
    total_e_t = torch.from_numpy(total_e.reshape(1, 1)).float().to(DEVICE)
    C_d_t = torch.from_numpy(C_d_mask[np.newaxis, :, :]).float().to(DEVICE)  # (1, P, 341)
    H_d_t = torch.from_numpy(H_d_mask[np.newaxis, :, :]).float().to(DEVICE)
    N_d_t = torch.from_numpy(N_d_mask[np.newaxis, :, :]).float().to(DEVICE)
    O_d_t = torch.from_numpy(O_d_mask[np.newaxis, :, :]).float().to(DEVICE)

    dos_model.eval()
    with torch.no_grad():
        norm_dos_t, vbcb_t = dos_model(
            X_C_t2, X_H_t2, X_N_t2, X_O_t2,
            total_e_t,
            C_d_t, H_d_t, N_d_t, O_d_t,
        )

    norm_dos = norm_dos_t.cpu().numpy().squeeze(0)          # (343,)
    vbcb = vbcb_t.cpu().numpy().squeeze(0)                  # (2,)
    VB = -vbcb[0]
    CB = -vbcb[1]
    BG = CB - VB

    # 可选: 保存 DOS 文件
    if args.plot_dos:
        import matplotlib.pyplot as plt
        energy_grid = np.arange(-33.0, 1.1, 0.1, dtype=np.float32)
        plt.plot(energy_grid, norm_dos, "r-", linewidth=1)
        plt.axvline(VB, color="b", linestyle=":")
        plt.axvline(CB, color="g", linewidth=1)
        plt.axvline(0, color="k", linestyle="--", linewidth=2)
        plt.xlabel("Energy (eV)")
        plt.ylabel("DOS")
        plt.tight_layout()
        plt.savefig(os.path.join("dos_plot_" + os.path.basename(folder) + ".png"), dpi=300)
        plt.clf()

    energy_grid = np.arange(-33.0, 1.1, 0.1, dtype=np.float32)
    return energy_grid, norm_dos, VB, CB, BG, None  # “None” 表示暂不输出不确定度


# ------------------------------------------------------------------------------
# 导出给 CLI 使用
# ------------------------------------------------------------------------------
__all__ = [
    "init_DOSmod",
    "load_pretrained_dos_model",
    "train_dos_model",
    "infer_dos",
]
