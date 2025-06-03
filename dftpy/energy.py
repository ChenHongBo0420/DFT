# dftpy/energy.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .fp import fp_atom, fp_chg_norm, fp_norm
from .data_io import get_efp_data, pad_efp_data
from pathlib import Path
from pymatgen.io.vasp.outputs import Poscar

# dftpy/energy.py 顶部
__all__ = [
    "train_energy_model",
    "load_pretrained_energy_model",
    "infer_energy",
]

# ------------------------------------------------------------------------------
# Constants and device
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Utility: read energies, forces, and pressures from files
# ------------------------------------------------------------------------------
def e_train(folder: str, tot_atoms: int):
    """
    从单个文件夹读取 DFT 参考的能量、力和应力：
      - 文件 “energy” 包含体系总能量（按原子归一化前）
      - 文件 “forces” 包含每个原子的力向量（n_atoms × 3）
      - 文件 “stress” 包含 6 个应力分量

    返回：
      - Energy_norm: np.ndarray, shape=(1,), 归一化后总能量 = |Energy_raw|/tot_atoms
      - forces:     np.ndarray, shape=(n_atoms, 3)
      - pressure:   np.ndarray, shape=(1, 6)
    """
    # 1. 读取能量并归一化
    with open(os.path.join(folder, "energy")) as f:
        lines = f.readlines()
    levels = [float(x) for x in lines[0].split()]
    Energy_norm = np.abs(np.array(levels)) / tot_atoms  # shape=(1,)

    # 2. 读取力
    forces_data = np.loadtxt(os.path.join(folder, "forces"), dtype=np.float32)  # shape=(n_atoms, 3)

    # 3. 读取应力（6 分量）
    press_data = np.loadtxt(os.path.join(folder, "stress"), dtype=np.float32)   # 6 floats
    press = press_data.reshape(1, 6)  # shape=(1,6)

    return Energy_norm.astype(np.float32), forces_data.astype(np.float32), press.astype(np.float32)


# ------------------------------------------------------------------------------
# Single-atom network (shared by C, N, O) and a lighter variant for H
# ------------------------------------------------------------------------------
class SingleAtomNet(nn.Module):
    """
    单原子网络：输入指纹 + 基函数 → 输出 10 维向量：
      (E, f_x, f_y, f_z, σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 10)  # 输出 10 维
        )

    def forward(self, x):
        # x: (batch_size * padding_size, input_dim)
        return self.net(x)  # (batch_size * padding_size, 10)


# ------------------------------------------------------------------------------
# EnergyModel: 综合 C/H/N/O 四类子网处理整个结构
# ------------------------------------------------------------------------------
class EnergyModel(nn.Module):
    """
    结构能量模型，将四类原子的指纹和基函数输入，输出：
      - E_tot:    (batch, 1)
      - forcesC:  (batch, nC, 3)
      - forcesH:  (batch, nH, 3)
      - forcesN:  (batch, nN, 3)
      - forcesO:  (batch, nO, 3)
      - Press:    (batch, 6)
    注意：为了保持与原 Keras 代码一致，这里所有输入都 pad 到相同 padding_size，额外传入掩码与原子个数。
    """
    def __init__(self, dim_C: int, dim_H: int, dim_N: int, dim_O: int, padding_size: int):
        super().__init__()
        self.padding_size = padding_size

        # 子网：C/N/O 三类使用同样维度（orig 709），H 使用 577
        self.netC = SingleAtomNet(dim_C)  # for C and N and O
        self.netH = SingleAtomNet(dim_H)

    def forward(
        self,
        X_C,  # (batch, padding_size, dim_C)
        X_H,  # (batch, padding_size, dim_H)
        X_N,  # (batch, padding_size, dim_C)
        X_O,  # (batch, padding_size, dim_C)
        num_atoms,  # (batch, 1)
        C_m, H_m, N_m, O_m  # each: (batch, padding_size, 1) mask 用于按元素算能量贡献
    ):
        batch_size = X_C.size(0)
        P = self.padding_size

        # Flatten batch 与 padding 维度，送入子网
        # C 类
        C_in = X_C.reshape(batch_size * P, -1)  # (B*P, dim_C)
        outC = self.netC(C_in)                  # (B*P, 10)
        outC = outC.reshape(batch_size, P, 10)  # (B, P, 10)

        # H 类
        H_in = X_H.reshape(batch_size * P, -1)  # (B*P, dim_H)
        outH = self.netH(H_in)                  # (B*P, 10)
        outH = outH.reshape(batch_size, P, 10)

        # N 类
        N_in = X_N.reshape(batch_size * P, -1)
        outN = self.netC(N_in)                  # 使用同 netC
        outN = outN.reshape(batch_size, P, 10)

        # O 类
        O_in = X_O.reshape(batch_size * P, -1)
        outO = self.netC(O_in)                  # 使用同 netC
        outO = outO.reshape(batch_size, P, 10)

        # 对每个元素类别分别拆分
        # E (1), forces (3), press (6) → indices [0], [1:4], [4:10]
        E_C    = torch.abs(outC[..., 0:1])      # (B, P, 1)
        fC     = outC[..., 1:4]                 # (B, P, 3)
        pC = outC[..., 4:10]                    # (B, P, 6)

        E_H    = torch.abs(outH[..., 0:1])
        fH     = outH[..., 1:4]
        pH = outH[..., 4:10]

        E_N    = torch.abs(outN[..., 0:1])
        fN     = outN[..., 1:4]
        pN = outN[..., 4:10]

        E_O    = torch.abs(outO[..., 0:1])
        fO     = outO[..., 1:4]
        pO = outO[..., 4:10]

        # 只保留 mask 掩码指定的有效原子条目
        E_C = E_C * C_m      # (B, P, 1)
        E_H = E_H * H_m
        E_N = E_N * N_m
        E_O = E_O * O_m

        fC = fC * C_m        # 广播 (B, P, 3)
        fH = fH * H_m
        fN = fN * N_m
        fO = fO * O_m

        pC_masked = pC * C_m  # (B, P, 6)
        pH_masked = pH * H_m
        pN_masked = pN * N_m
        pO_masked = pO * O_m

        # 求和：先 sum over atoms维度 (axis=1)
        # Sum(E_i) → (B, 1), Sum(press_i) → (B, 6)
        sum_E_C = torch.sum(E_C, dim=1, keepdim=True)  # (B, 1)
        sum_E_H = torch.sum(E_H, dim=1, keepdim=True)
        sum_E_N = torch.sum(E_N, dim=1, keepdim=True)
        sum_E_O = torch.sum(E_O, dim=1, keepdim=True)

        sum_E_all = sum_E_C + sum_E_H + sum_E_N + sum_E_O  # (B, 1)

        # 按原子数归一化
        E_tot = sum_E_all / num_atoms  # (B, 1)

        sum_pC = torch.sum(pC_masked, dim=1)  # (B, 6)
        sum_pH = torch.sum(pH_masked, dim=1)
        sum_pN = torch.sum(pN_masked, dim=1)
        sum_pO = torch.sum(pO_masked, dim=1)

        Press = (sum_pC + sum_pH + sum_pN + sum_pO) / num_atoms  # (B, 6)

        # Forces: 保留对每个元素的按原子输出 (batch, P, 3)
        # 但后续预测时会只取前 at_elem[i] 条记录，并 concat
        return E_tot, fC, fH, fN, fO, Press


# ------------------------------------------------------------------------------
# Instantiate model given padding_size and input feature dims
# ------------------------------------------------------------------------------
def init_Emod(padding_size: int, dim_C: int, dim_H: int, dim_N: int, dim_O: int):
    model = EnergyModel(dim_C, dim_H, dim_N, dim_O, padding_size).to(DEVICE)
    return model

# ------------------------------------------------------------------------------
# Load or freeze weights
# ------------------------------------------------------------------------------
def model_weights(train_e: bool, new_weights_e: bool, model_E: nn.Module, checkpoint_path: str):
    """
    根据 train_e 和 new_weights_e 选择加载哪个权重文件：
      - 如果 train_e 或 new_weights_e 为 True，加载 'newEmodel.pth'
      - 否则加载已训练好的 CONFIG_PATH
    """
    if train_e or new_weights_e:
        path = "newEmodel.pth"
    else:
        path = checkpoint_path

    if os.path.exists(path):
        model_E.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"无法找到模型权重文件: {path}")


# ------------------------------------------------------------------------------
# 预测接口：energy_predict
# ------------------------------------------------------------------------------
def energy_predict(
    X_C_np: np.ndarray,
    X_H_np: np.ndarray,
    X_N_np: np.ndarray,
    X_O_np: np.ndarray,
    basis1_np: np.ndarray,
    basis2_np: np.ndarray,
    basis3_np: np.ndarray,
    basis4_np: np.ndarray,
    C_m_np: np.ndarray,
    H_m_np: np.ndarray,
    N_m_np: np.ndarray,
    O_m_np: np.ndarray,
    num_atoms_np: np.ndarray,
    model_E: nn.Module,
    train_e: bool,
    new_weights_e: bool,
    checkpoint_path: str,
):
    """
    PyTorch 版预测能量、力和应力。
    输入：
      - X_?_np:   np.ndarray, 每个 shape = (1, padding_size, feat_dim_?)
      - basis?_np: np.ndarray, shape = (1, padding_size, 9)
      - C_m_np etc: np.ndarray, shape = (1, padding_size, 1) 掩码
      - num_atoms_np: np.ndarray, shape = (1, 1)
      - model_E: 能量模型实例
      - train_e, new_weights_e: booleans, 决定加载哪个权重
      - checkpoint_path: 预训练权重路径

    返回：
      - Pred_Energy: float 标量
      - ForC, ForH, ForN, ForO: np.ndarray, shapes=(nC, 3), (nH,3), (nN,3), (nO,3)
      - pred_press: np.ndarray, shape=(6,)
    """
    # 1. 转 NumPy 到 Torch Tensor
    X_C = torch.from_numpy(X_C_np).float().to(DEVICE)
    X_H = torch.from_numpy(X_H_np).float().to(DEVICE)
    X_N = torch.from_numpy(X_N_np).float().to(DEVICE)
    X_O = torch.from_numpy(X_O_np).float().to(DEVICE)

    C_m = torch.from_numpy(C_m_np).float().to(DEVICE)
    H_m = torch.from_numpy(H_m_np).float().to(DEVICE)
    N_m = torch.from_numpy(N_m_np).float().to(DEVICE)
    O_m = torch.from_numpy(O_m_np).float().to(DEVICE)

    num_atoms = torch.from_numpy(num_atoms_np).float().to(DEVICE)  # shape=(1,1)

    # 2. 拼接指纹与基函数：BASE = np.concatenate((X_?, basis?), axis=-1)
    #    因为原 Keras 代码在预测前拼接了一次
    X_C_cat = torch.cat([X_C, torch.from_numpy(basis1_np).float().to(DEVICE)], dim=-1)  # (1, P, dim_C+9)
    X_H_cat = torch.cat([X_H, torch.from_numpy(basis2_np).float().to(DEVICE)], dim=-1)
    X_N_cat = torch.cat([X_N, torch.from_numpy(basis3_np).float().to(DEVICE)], dim=-1)
    X_O_cat = torch.cat([X_O, torch.from_numpy(basis4_np).float().to(DEVICE)], dim=-1)

    # 3. 加载权重
    model_weights(train_e, new_weights_e, model_E, checkpoint_path)
    model_E.eval()

    # 4. 推理
    with torch.no_grad():
        E_tot, fC, fH, fN, fO, Press = model_E(
            X_C_cat, X_H_cat, X_N_cat, X_O_cat, num_atoms, C_m, H_m, N_m, O_m
        )

    # 转回 NumPy
    Pred_Energy = (-1.0) * E_tot.cpu().numpy()  # Keras 代码对 E 乘以 -1
    fC_np = fC.cpu().numpy().squeeze(0)  # (padding_size, 3)
    fH_np = fH.cpu().numpy().squeeze(0)
    fN_np = fN.cpu().numpy().squeeze(0)
    fO_np = fO.cpu().numpy().squeeze(0)
    pred_press = Press.cpu().numpy().squeeze(0)  # (6,)

    return Pred_Energy.item(), fC_np, fH_np, fN_np, fO_np, pred_press


# ------------------------------------------------------------------------------
# PyTorch Dataset for energy training
# ------------------------------------------------------------------------------
class EnergyDataset(Dataset):
    def __init__(
        self,
        X_C, X_H, X_N, X_O,
        C_m, H_m, N_m, O_m, basis1, basis2, basis3, basis4,
        X_at, ener_ref, fC, fH, fN, fO, press_ref
    ):
        """
        传入所有训练/验证数据的 NumPy 数组，构建 Dataset。
        每个样本由以下部分组成：
          - X_C[i], X_H[i], X_N[i], X_O[i]: (padding_size, feat_dim_?)
          - C_m[i], H_m[i], N_m[i], O_m[i]: (padding_size, 1)
          - basis?_i： (padding_size, 9)
          - X_at[i]: (1,) 原子总数
          - ener_ref[i]: (1,) 参考能量
          - fC[i], fH[i], fN[i], fO[i]: (padding_size, 3) 参考力，pad 部分用 1000 填充
          - press_ref[i]: (6,)
        """
        self.X_C = torch.from_numpy(X_C).float()
        self.X_H = torch.from_numpy(X_H).float()
        self.X_N = torch.from_numpy(X_N).float()
        self.X_O = torch.from_numpy(X_O).float()
        self.C_m = torch.from_numpy(C_m).float()
        self.H_m = torch.from_numpy(H_m).float()
        self.N_m = torch.from_numpy(N_m).float()
        self.O_m = torch.from_numpy(O_m).float()
        self.basis1 = torch.from_numpy(basis1).float()
        self.basis2 = torch.from_numpy(basis2).float()
        self.basis3 = torch.from_numpy(basis3).float()
        self.basis4 = torch.from_numpy(basis4).float()

        self.X_at = torch.from_numpy(X_at).float()             # (n_samples,1)
        self.ener_ref = torch.from_numpy(ener_ref).float()     # (n_samples,1)
        self.fC = torch.from_numpy(fC).float()                 # (n_samples, padding_size, 3)
        self.fH = torch.from_numpy(fH).float()
        self.fN = torch.from_numpy(fN).float()
        self.fO = torch.from_numpy(fO).float()
        self.press_ref = torch.from_numpy(press_ref).float()   # (n_samples,6)

    def __len__(self):
        return self.X_C.shape[0]

    def __getitem__(self, idx):
        return (
            self.X_C[idx], self.X_H[idx], self.X_N[idx], self.X_O[idx],
            self.C_m[idx], self.H_m[idx], self.N_m[idx], self.O_m[idx],
            self.basis1[idx], self.basis2[idx], self.basis3[idx], self.basis4[idx],
            self.X_at[idx],
            self.ener_ref[idx],
            self.fC[idx], self.fH[idx], self.fN[idx], self.fO[idx],
            self.press_ref[idx]
        )


# ------------------------------------------------------------------------------
# Custom loss for forces: 忽略 pad 部分（值为 1000 的位置）
# ------------------------------------------------------------------------------
def masked_mse_loss(y_pred, y_true):
    """
    输入：y_pred, y_true 均形状 (batch, padding_size, 3)
    使用掩码：当 y_true == 1000 时忽略该原子条目。
    """
    mask = (y_true != 1000.0).float()  # (B, P, 3)，pad 部分为0，其它为1
    diff = (y_true - y_pred) * mask
    mse = diff.pow(2).sum() / mask.sum().clamp(min=1.0)
    return mse


# ------------------------------------------------------------------------------
# Training / Retrain pipeline
# ------------------------------------------------------------------------------
def retrain_emodel(
    X_C, X_H, X_N, X_O,
    C_m, H_m, N_m, O_m,
    basis1, basis2, basis3, basis4,
    X_at, ener_ref, fC, fH, fN, fO, press_ref,
    X_val_C, X_val_H, X_val_N, X_val_O,
    C_mV, H_mV, N_mV, O_mV,
    basis1V, basis2V, basis3V, basis4V,
    X_at_val, ener_val, fCV, fHV, fNV, fOV, press_val,
    epochs: int,
    batch_size: int,
    patience: int,
    padding_size: int,
    dim_C: int = 709,
    dim_H: int = 577,
    dim_N: int = 709,
    dim_O: int = 709,
    checkpoint_path: str = "best_emodel.pth"
):
    """
    在 PyTorch 中训练能量模型，流程：
      1. 构造 DataLoader（训练 + 验证）
      2. 定义模型、优化器、损失函数和早停
      3. 迭代训练，验证集 early stopping，保存最佳模型权重到 newEmodel.pth
    """
    # 1. 创建 Dataset 和 DataLoader
    train_dataset = EnergyDataset(
        X_C, X_H, X_N, X_O, C_m, H_m, N_m, O_m,
        basis1, basis2, basis3, basis4,
        X_at, ener_ref, fC, fH, fN, fO, press_ref
    )
    val_dataset = EnergyDataset(
        X_val_C, X_val_H, X_val_N, X_val_O, C_mV, H_mV, N_mV, O_mV,
        basis1V, basis2V, basis3V, basis4V,
        X_at_val, ener_val, fCV, fHV, fNV, fOV, press_val
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. 初始化模型
    model = init_Emod(padding_size, dim_C, dim_H, dim_N, dim_O).to(DEVICE)

    # 3. 定义损失函数和优化器
    criterion_energy = nn.MSELoss()
    criterion_force  = masked_mse_loss
    criterion_press  = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    best_epoch = 0

    # 4. 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            (
                X_C_b, X_H_b, X_N_b, X_O_b,
                C_m_b, H_m_b, N_m_b, O_m_b,
                basis1_b, basis2_b, basis3_b, basis4_b,
                X_at_b, ener_b, fC_b, fH_b, fN_b, fO_b, press_b
            ) = batch

            # 转到 DEVICE
            X_C_b = X_C_b.to(DEVICE)
            X_H_b = X_H_b.to(DEVICE)
            X_N_b = X_N_b.to(DEVICE)
            X_O_b = X_O_b.to(DEVICE)
            C_m_b = C_m_b.to(DEVICE)
            H_m_b = H_m_b.to(DEVICE)
            N_m_b = N_m_b.to(DEVICE)
            O_m_b = O_m_b.to(DEVICE)
            basis1_b = basis1_b.to(DEVICE)
            basis2_b = basis2_b.to(DEVICE)
            basis3_b = basis3_b.to(DEVICE)
            basis4_b = basis4_b.to(DEVICE)
            X_at_b = X_at_b.to(DEVICE)
            ener_b = ener_b.to(DEVICE)
            fC_b = fC_b.to(DEVICE)
            fH_b = fH_b.to(DEVICE)
            fN_b = fN_b.to(DEVICE)
            fO_b = fO_b.to(DEVICE)
            press_b = press_b.to(DEVICE)

            # 拼接指纹与基函数
            X_C_cat = torch.cat([X_C_b, basis1_b], dim=-1)
            X_H_cat = torch.cat([X_H_b, basis2_b], dim=-1)
            X_N_cat = torch.cat([X_N_b, basis3_b], dim=-1)
            X_O_cat = torch.cat([X_O_b, basis4_b], dim=-1)

            optimizer.zero_grad()
            E_pred, fC_pred, fH_pred, fN_pred, fO_pred, p_pred = model(
                X_C_cat, X_H_cat, X_N_cat, X_O_cat, X_at_b, C_m_b, H_m_b, N_m_b, O_m_b
            )

            # 计算多任务损失
            loss_E = criterion_energy(E_pred, ener_b) * 1000.0
            loss_fC = criterion_force(fC_pred, fC_b) * 10.0
            loss_fH = criterion_force(fH_pred, fH_b) * 10.0
            loss_fN = criterion_force(fN_pred, fN_b) * 10.0
            loss_fO = criterion_force(fO_pred, fO_b) * 10.0
            loss_P = criterion_press(p_pred, press_b) * 0.1

            loss = loss_E + loss_fC + loss_fH + loss_fN + loss_fO + loss_P
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (
                    X_C_b, X_H_b, X_N_b, X_O_b,
                    C_m_b, H_m_b, N_m_b, O_m_b,
                    basis1_b, basis2_b, basis3_b, basis4_b,
                    X_at_b, ener_b, fC_b, fH_b, fN_b, fO_b, press_b
                ) = batch

                X_C_b = X_C_b.to(DEVICE)
                X_H_b = X_H_b.to(DEVICE)
                X_N_b = X_N_b.to(DEVICE)
                X_O_b = X_O_b.to(DEVICE)
                C_m_b = C_m_b.to(DEVICE)
                H_m_b = H_m_b.to(DEVICE)
                N_m_b = N_m_b.to(DEVICE)
                O_m_b = O_m_b.to(DEVICE)
                basis1_b = basis1_b.to(DEVICE)
                basis2_b = basis2_b.to(DEVICE)
                basis3_b = basis3_b.to(DEVICE)
                basis4_b = basis4_b.to(DEVICE)
                X_at_b = X_at_b.to(DEVICE)
                ener_b = ener_b.to(DEVICE)
                fC_b = fC_b.to(DEVICE)
                fH_b = fH_b.to(DEVICE)
                fN_b = fN_b.to(DEVICE)
                fO_b = fO_b.to(DEVICE)
                press_b = press_b.to(DEVICE)

                X_C_cat = torch.cat([X_C_b, basis1_b], dim=-1)
                X_H_cat = torch.cat([X_H_b, basis2_b], dim=-1)
                X_N_cat = torch.cat([X_N_b, basis3_b], dim=-1)
                X_O_cat = torch.cat([X_O_b, basis4_b], dim=-1)

                E_pred, fC_pred, fH_pred, fN_pred, fO_pred, p_pred = model(
                    X_C_cat, X_H_cat, X_N_cat, X_O_cat, X_at_b, C_m_b, H_m_b, N_m_b, O_m_b
                )

                loss_E = criterion_energy(E_pred, ener_b) * 1000.0
                loss_fC = criterion_force(fC_pred, fC_b) * 10.0
                loss_fH = criterion_force(fH_pred, fH_b) * 10.0
                loss_fN = criterion_force(fN_pred, fN_b) * 10.0
                loss_fO = criterion_force(fO_pred, fO_b) * 10.0
                loss_P = criterion_press(p_pred, press_b) * 0.1

                val_loss += (loss_E + loss_fC + loss_fH + loss_fN + loss_fO + loss_P).item()

        # 记录并早停
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            # 保存最优模型
            torch.save(model.state_dict(), "newEmodel.pth")

        if epoch - best_epoch >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Val Loss = {best_val_loss:.6f} at epoch {best_epoch}.")


# ------------------------------------------------------------------------------
# Wrapper to load pretrained energy model (只用于推理)
# ------------------------------------------------------------------------------
def load_pretrained_energy_model(checkpoint_path: str, padding_size: int, dim_C: int = 709, dim_H: int = 577, dim_N: int = 709, dim_O: int = 709):
    """
    加载一个训练好的能量模型并返回。
    """
    model = init_Emod(padding_size, dim_C, dim_H, dim_N, dim_O)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"能量模型权重文件未找到: {checkpoint_path}")


# ------------------------------------------------------------------------------
# Infer interface for CLI: infer_energy
# ------------------------------------------------------------------------------
def infer_energy(
    folder: str,
    chg_model,    # 不直接使用，可传 None
    energy_model: nn.Module,
    padding_size: int,
    args
):
    """
    CLI 中调用的推理函数：
      - folder: 单个结构文件夹路径
      - chg_model: 电荷模型（这里不使用，因为推理前已经在 fp_chg_norm 里得到所有必要的特征）
      - energy_model: 已加载的 PyTorch 能量模型
      - padding_size: 填充尺寸
      - args: 命令行参数（用于读取 grid_spacing, cut_off_rad, 等）

    返回：
      - total_energy: float
      - forces: np.ndarray, shape=(num_atoms, 3) 总力按原子顺序排列
      - stress: np.ndarray, shape=(6,)
    """
    # 1. 读取 POSCAR，生成指纹 + 基函数 + 掩码
    poscar = Poscar.from_file(os.path.join(folder, "POSCAR"))
    struct = poscar.structure

    # 调用 fp.py 中的 fp_atom + 后续分割逻辑
    from .fp import fp_atom, fp_chg_norm, fp_norm
    # 1.1 生成纯指纹和基函数
    dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(
        struct,
        args.grid_spacing,
        args.cut_off_rad,
        args.widest_gaussian,
        args.narrowest_gaussian,
        args.num_gamma
    )
    # at_elem = [nC, nH, nN, nO]
    # 创建 padding 到 padding_size
    # 拆分 per-element 再 pad
    from .data_io import chg_data
    X_3D1, X_3D2, X_3D3, X_3D4, basis1, basis2, basis3, basis4, C_m, H_m, N_m, O_m = chg_data(
        dset, basis_mat, at_elem[0], at_elem[1], at_elem[2], at_elem[3], padding_size
    )

    # 1.2 调用电荷模型预测电荷系数
    Coef1, Coef2, Coef3, Coef4, _, _, _, _ = chg_model.predict(  # 假设 chg_model 兼容此接口
        X_3D1, X_3D2, X_3D3, X_3D4, at_elem[0], at_elem[1], at_elem[2], at_elem[3], sites_elem, chg_model, at_elem
    )
    # 用 fp_chg_norm 得到带电荷维的指纹
    scaler_paths = (
        os.path.join(os.path.dirname(__file__), "scalers/Scale_model_C.joblib"),
        os.path.join(os.path.dirname(__file__), "scalers/Scale_model_H.joblib"),
        os.path.join(os.path.dirname(__file__), "scalers/Scale_model_N.joblib"),
        os.path.join(os.path.dirname(__file__), "scalers/Scale_model_O.joblib"),
    )
    X_C, X_H, X_N, X_O = fp_chg_norm(
        Coef1, Coef2, Coef3, Coef4,
        X_3D1, X_3D2, X_3D3, X_3D4,
        padding_size,
        scaler_paths
    )

    # 1.3 最后做一次归一化
    X_C, X_H, X_N, X_O = fp_norm(X_C, X_H, X_N, X_O, padding_size, scaler_paths)

    # 2. 调用 energy_predict
    # basis1~4, C_m~O_m 都已是 NumPy，num_atoms 传 (1,1)
    num_atoms_np = np.array([[num_atoms]], dtype=np.float32)
    energy, fC, fH, fN, fO, stress = energy_predict(
        X_C, X_H, X_N, X_O,
        basis1, basis2, basis3, basis4,
        C_m, H_m, N_m, O_m,
        num_atoms_np,
        energy_model,
        args.train_e, args.new_weights_e,
        os.path.join(os.path.dirname(__file__), "../Trained_models/weights_EFP.pth")
    )

    # 3. 合并 forces 按原子顺序：C->H->N->O
    forces_list = []
    if at_elem[0] > 0:
        forces_list.append(fC[: at_elem[0], :])
    if at_elem[1] > 0:
        forces_list.append(fH[: at_elem[1], :])
    if at_elem[2] > 0:
        forces_list.append(fN[: at_elem[2], :])
    if at_elem[3] > 0:
        forces_list.append(fO[: at_elem[3], :])

    forces = np.vstack(forces_list)  # shape=(num_atoms,3)

    return energy, forces, stress

def train_energy_model(train_folders, val_folders, chg_model, padding_size, args):
    """
    1) 用 get_efp_data 读训练/验证集原始数据
    2) 用 pad_efp_data 把 fingerprint/basis/forces/pressure pad 到统一大小
    3) 用 fp_norm 对 fingerprint 做归一化
    4) 调用 retrain_emodel 进行真正的 PyTorch 训练
    """

    # ──── 1) 读入训练/验证数据 ────
    ener_ref, forces_pre, press_ref, X_pre, basis_pre, X_at, X_el, X_elem = get_efp_data(train_folders)
    ener_val, forces_val, press_val, X_val_pre, basis_val_pre, X_at_val, X_el_val, X_elem_val = get_efp_data(val_folders)

    padding_size = int(padding_size)  # ensure int

    # ──── 2) pad_efp_data: 把 fingerprint/basis/forces pad 到统一 (padding_size) ────
    forces1, forces2, forces3, forces4, \
    X_1, X_2, X_3, X_4, \
    basis1, basis2, basis3, basis4, \
    C_m, H_m, N_m, O_m = pad_efp_data(X_elem, X_pre, forces_pre, basis_pre, padding_size)

    forcesV1, forcesV2, forcesV3, forcesV4, \
    X_1V, X_2V, X_3V, X_4V, \
    basis1V, basis2V, basis3V, basis4V, \
    C_mV, H_mV, N_mV, O_mV = pad_efp_data(X_elem_val, X_val_pre, forces_val, basis_val_pre, padding_size)

    # ──── 3) 拼出四个 scaler_paths 的绝对路径 ────
    base_dir = Path(__file__).parent  # dftpy/ 目录
    scaler_paths = (
        str(base_dir / "scalers" / "Scale_model_C.joblib"),
        str(base_dir / "scalers" / "Scale_model_H.joblib"),
        str(base_dir / "scalers" / "Scale_model_N.joblib"),
        str(base_dir / "scalers" / "Scale_model_O.joblib"),
    )

    # 用 fp_norm 对 fingerprint 做 MaxAbsScaler 归一化
    X_C, X_H, X_N, X_O   = fp_norm(X_1, X_2, X_3, X_4, padding_size, scaler_paths)
    X_CV, X_HV, X_NV, X_OV = fp_norm(X_1V, X_2V, X_3V, X_4V, padding_size, scaler_paths)

    # ──── 4) 自动计算 dim_C, dim_H, dim_N, dim_O ────
    #    fingerprint dim = X_C.shape[-1]  （譬如 360）
    #    basis dim       = basis1.shape[-1]  （即 9）
    #    所以网络的输入维度 = fingerprint_dim + basis_dim = 360 + 9 = 369
    fingerprint_dim = X_C.shape[-1]  # 举例是 360
    basis_dim       = basis1.shape[-1]  # 9

    dim_C = fingerprint_dim + basis_dim
    dim_H = fingerprint_dim + basis_dim
    dim_N = fingerprint_dim + basis_dim
    dim_O = fingerprint_dim + basis_dim

    # ──── 5) 调用 retrain_emodel，传入上面算好的 dim_C/… ────
    retrain_emodel(
        # 训练集所有输入
        X_C, X_H, X_N, X_O,
        C_m, H_m, N_m, O_m,
        basis1, basis2, basis3, basis4,
        X_at, ener_ref, forces1, forces2, forces3, forces4, press_ref,
        # 验证集所有输入
        X_CV, X_HV, X_NV, X_OV,
        C_mV, H_mV, N_mV, O_mV,
        basis1V, basis2V, basis3V, basis4V,
        X_at_val, ener_val, forcesV1, forcesV2, forcesV3, forcesV4, press_val,

        # 训练配置
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        patience    = args.patience,
        padding_size= padding_size,

        # **把网络输入维度改成 “360 + 9 = 369”**（而非以前写死的 709/577/709/709）
        dim_C = dim_C,
        dim_H = dim_H,
        dim_N = dim_N,
        dim_O = dim_O,

        checkpoint_path = "best_emodel.pth"
    )

    print("Energy model training done. Best weights saved as newEmodel.pth (in current working dir).")
