# dftpy/data_io.py

import os
import numpy as np
import pandas as pd
from pymatgen.io.vasp.outputs import Poscar

# Mapping from atomic number to valence electrons
elec_dict = {6: 4, 1: 1, 7: 5, 8: 6}


def read_file_list(csv_path: str, col: str):
    """
    从 CSV 文件读取指定列，并返回非空的文件夹路径列表。
    """
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise KeyError(f"CSV 中未找到列 {col}")
    return df[col].dropna().astype(str).tolist()


def read_poscar(folder: str):
    """
    读取给定文件夹下的 POSCAR 文件，返回 pymatgen.Structure 对象。
    """
    poscar_path = os.path.join(folder.rstrip("/"), "POSCAR")
    return Poscar.from_file(poscar_path).structure


def save_charges(chg_vals: np.ndarray, filepath: str):
    """
    将一维电荷数组保存到文本文件（每行一个值）。
    """
    np.savetxt(filepath, chg_vals, fmt="%.6f")


def save_energy(energy: float, forces: np.ndarray, stress: np.ndarray, filepath: str):
    """
    将总能量、力和应力写入同一个文本文件。
    """
    with open(filepath, "w") as f:
        f.write(f"Total potential energy (eV): {energy:.6f}\n\n")
        f.write("Forces (eV/Å):\n")
        np.savetxt(f, forces, fmt="%.6f")
        f.write("\nStress tensor components (kB):\n")
        f.write(" ".join([f"{x:.6f}" for x in stress]) + "\n")


def save_dos(energy_grid: np.ndarray, dos_vals: np.ndarray, vb: float, cb: float, bg: float, filepath: str):
    """
    将 DOS 曲线和带边信息写入文本文件。
    - DOS 主文件：两列 (Energy, DOS)
    - 信息文件：带顶、带底、带隙等
    """
    # 保存 DOS 数值
    header = "Energy(eV)    DOS"
    data = np.column_stack((energy_grid, dos_vals))
    np.savetxt(filepath, data, fmt="%.6f", header=header)

    # 保存额外信息
    info_path = filepath.replace(".txt", "_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Valence band maximum (VB): {vb:.6f} eV\n")
        f.write(f"Conduction band minimum (CB): {cb:.6f} eV\n")
        f.write(f"Bandgap (BG): {bg:.6f} eV\n")


def get_max_atom_count(folders: list):
    """
    遍历所有给定文件夹，读取 POSCAR，返回这些结构中原子数的最大值。
    """
    max_cnt = 0
    for folder in folders:
        struct = read_poscar(folder)
        cnt = struct.num_sites
        max_cnt = max(max_cnt, cnt)
    return max_cnt


def pad_to(arr: np.ndarray, target_rows: int, pad_value=0.0):
    """
    将二维数组 arr（shape = (n_rows, n_feats)）沿第 0 轴 pad 到 (target_rows, n_feats)。
    若 n_rows < target_rows，则在底部追加 pad_value；否则不剪裁直接返回原数组。
    """
    n_rows, n_feats = arr.shape
    if n_rows >= target_rows:
        return arr.copy()
    pad_amount = target_rows - n_rows
    pad_block = np.full((pad_amount, n_feats), pad_value, dtype=arr.dtype)
    return np.vstack([arr, pad_block])


def get_def_data(folder: str):
    """
    读取单个文件夹中的 POSCAR 并返回：
        vol           : 晶胞体积 (float)
        supercell     : pymatgen.Structure
        dim           : 晶格矩阵 (3×3 array)
        total_elec    : 总价电子数 (int)
        elems_list    : 元素种类列表 (e.g. ["C","H","O"])
        poscar_data   : Poscar 对象（方便后续调用 .structure）
    """
    poscar_file = os.path.join(folder.rstrip("/"), "POSCAR")
    poscar_data = Poscar.from_file(poscar_file)
    supercell = poscar_data.structure
    vol = supercell.volume
    dim = supercell.lattice.matrix
    atoms = supercell.num_sites
    elems_list = sorted(list(set(poscar_data.site_symbols)))
    electrons_list = [elec_dict[x] for x in supercell.atomic_numbers]
    total_elec = sum(electrons_list)
    return vol, supercell, dim, total_elec, elems_list, poscar_data


def get_fp_all(at_elem: list, X_tot: np.ndarray, padding_size: int):
    """
    对所有元素类别的 “原子指纹向量 (X_tot)” 做 pad，并拼接：
    - at_elem: [i1, i2, i3, i4] 分别是 C/H/N/O 原子数
    - X_tot: shape = (i_total, 360)，即所有原子的 360 维指纹
    - 返回 X_pad: shape = (padding_size, 360 * 4)，按元素顺序拼接 pad 后的二维指纹
    """
    # 把 X_tot 拆为四段
    i1, i2, i3, i4 = at_elem
    offsets = [0, i1, i1 + i2, i1 + i2 + i3]
    X_npad = []

    # 对每个元素类别，slice 出相应行，再 pad 到 (padding_size, 360)
    for idx, count in enumerate([i1, i2, i3, i4]):
        if count > 0:
            arr = X_tot[offsets[idx]: offsets[idx] + count, :360]  # shape=(count,360)
        else:
            arr = np.zeros((1, 360), dtype=np.float32)
        arr_padded = pad_to(arr, padding_size, pad_value=0.0)  # (padding_size,360)
        X_npad.append(arr_padded)

    # 拼接 4 段 (padding_size, 360) → (padding_size, 360*4)
    X_pad = np.concatenate(X_npad, axis=1)
    return X_pad


def get_fp_basis_F(
    at_elem: list,
    X_tot: np.ndarray,
    forces_data: np.ndarray,
    base_mat: np.ndarray,
    padding_size: int
):
    """
    对所有元素类别做 pad，同时返回：
    - X_pad: 指纹 pad 到 (padding_size, 360)
    - basis_pad: 基函数 pad 到 (padding_size, 9)
    - forces_pad: 力向量 pad 到 (padding_size, 3)
    """
    i1, i2, i3, i4 = at_elem
    offsets = [0, i1, i1 + i2, i1 + i2 + i3]
    X_npad, basis_npad, forces_npad = [], [], []

    counts = [i1, i2, i3, i4]
    for idx, count in enumerate(counts):
        if count > 0:
            arr = X_tot[offsets[idx]: offsets[idx] + count, :360]         # shape=(count,360)
            basis = base_mat[offsets[idx]: offsets[idx] + count].reshape(count, 9)  # shape=(count,9)
            forces = forces_data[offsets[idx]: offsets[idx] + count]       # shape=(count,3)
        else:
            arr = np.zeros((1, 360), dtype=np.float32)
            basis = np.zeros((1, 9), dtype=np.float32)
            forces = np.zeros((1, 3), dtype=np.float32)

        X_p = pad_to(arr, padding_size, pad_value=0.0)        # (padding_size,360)
        b_p = pad_to(basis, padding_size, pad_value=0.0)      # (padding_size,9)
        f_p = pad_to(forces, padding_size, pad_value=1000.0)  # (padding_size,3), 用大值 1000 作为 pad

        X_npad.append(X_p)
        basis_npad.append(b_p)
        forces_npad.append(f_p)

    # 把四个元素类别分别堆叠，然后拆分成回四份
    # X_npad: list of 4 arrays, each (padding_size,360)
    # basis_npad: list of 4 arrays, each (padding_size,9)
    # forces_npad: list of 4 arrays, each (padding_size,3)
    X_pad = np.concatenate(X_npad, axis=1)       # (padding_size, 360*4)
    basis_pad = np.concatenate(basis_npad, axis=1)   # (padding_size, 9*4)
    forces_pad = np.concatenate(forces_npad, axis=1) # (padding_size, 3*4)

    return X_pad, basis_pad, forces_pad


def get_all_data(data_list: list):
    """
    对一批文件夹执行“DEF 数据读取” + “电荷训练数据准备”：
    返回：
      X_1, X_2, X_3, X_4 : 每个样本的指纹按元素分片 (列表，每个元素是 2D array)
      Prop:              电荷标签 (VT stacked, shape=(n_samples_total, num_chg_bins))
      dataset_at1~4:     原子指纹（未 pad）按元素分类的列表
      X_at:              原子总数列表（垂直拼接成 col vector）
      X_el:              总电子数列表（垂直拼接成 col vector）

    用于后续 pad_efp_data 组合能量训练数据时用。
    """
    X_list_at1 = []
    X_list_at2 = []
    X_list_at3 = []
    X_list_at4 = []
    dataset_at1 = []
    dataset_at2 = []
    dataset_at3 = []
    dataset_at4 = []
    Prop_list = []
    At_list = []
    El_list = []

    for folder in data_list:
        vol, supercell, dim, total_elec, elems_list, poscar_data = get_def_data(folder)
        El_list.append([total_elec])

        # 生成“原子指纹”数据
        dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(poscar_data, supercell, elems_list)
        At_list.append([num_atoms])
        dataset1 = dset.copy()

        i1, i2, i3, i4 = at_elem
        # 按元素分割指纹数据
        X_at1 = dataset1[0:i1]
        X_at2 = dataset1[i1:i1 + i2]
        X_at3 = dataset1[i1 + i2:i1 + i2 + i3] if i3 > 0 else np.zeros((1, dataset1.shape[1]), dtype=np.float32)
        X_at4 = dataset1[i1 + i2 + i3:i1 + i2 + i3 + i4] if i4 > 0 else np.zeros((1, dataset1.shape[1]), dtype=np.float32)

        dataset_at1.append(X_at1)
        dataset_at2.append(X_at2)
        dataset_at3.append(X_at3)
        dataset_at4.append(X_at4)

        # 生成电荷训练相关数据
        chg, local_coords = chg_train(folder, vol, supercell, sites_elem, num_atoms, at_elem)
        num_chg_bins = chg.shape[0]
        dataset2 = local_coords.copy()
        # coef_predict 划分原始数据到 per-atom
        X_tot_at1, X_tot_at2, X_tot_at3, X_tot_at4 = chg_dat_prep(at_elem, dataset1, dataset2, i1, i2, i3, i4, num_chg_bins)

        Prop_list.append(chg.copy())
        X_list_at1.append(X_tot_at1.T)
        X_list_at2.append(X_tot_at2.T)
        X_list_at3.append(X_tot_at3.T)
        X_list_at4.append(X_tot_at4.T)

    X_1 = X_list_at1
    X_2 = X_list_at2
    X_3 = X_list_at3
    X_4 = X_list_at4
    Prop = np.vstack(Prop_list)
    X_at = np.vstack(At_list)
    X_el = np.vstack(El_list)
    return X_1, X_2, X_3, X_4, Prop, dataset_at1, dataset_at2, dataset_at3, dataset_at4, X_at, X_el


def chg_data(dataset1: np.ndarray, basis_mat: np.ndarray, i1: int, i2: int, i3: int, i4: int, padding_size: int):
    """
    类似原来 DFT.py 的 chg_data：
    - 接收未 pad 的 dataset1, basis_mat, per-element counts (i1,i2,i3,i4)
    - 返回  X_3D1~X_3D4: 每个元素类别的指纹 3D 张量 (1, padding_size, feature_dim)
               basis1~4: 对应基函数 3D 张量 (1, padding_size, 9)
               C_m, H_m, N_m, O_m: 掩码张量 (1, padding_size, 1)
    """
    # 拆分 per-element
    i1 = int(i1)
    i2 = int(i2)
    i3 = int(i3)
    i4 = int(i4)
    idx1 = 0
    idx2 = i1
    idx3 = i1 + i2
    idx4 = i1 + i2 + i3

    # C 类
    if i1 > 0:
        dat1 = dataset1[idx1:idx2]
        bas1 = basis_mat[idx1:idx2].reshape(i1, 9)
    else:
        dat1 = np.zeros((1, dataset1.shape[1]), dtype=np.float32)
        bas1 = np.zeros((1, 9), dtype=np.float32)

    # H 类
    if i2 > 0:
        dat2 = dataset1[idx2:idx3]
        bas2 = basis_mat[idx2:idx3].reshape(i2, 9)
    else:
        dat2 = np.zeros((1, dataset1.shape[1]), dtype=np.float32)
        bas2 = np.zeros((1, 9), dtype=np.float32)

    # N 类
    if i3 > 0:
        dat3 = dataset1[idx3:idx4]
        bas3 = basis_mat[idx3:idx4].reshape(i3, 9)
    else:
        dat3 = np.zeros((1, dataset1.shape[1]), dtype=np.float32)
        bas3 = np.zeros((1, 9), dtype=np.float32)

    # O 类
    if i4 > 0:
        dat4 = dataset1[idx4:idx4 + i4]
        bas4 = basis_mat[idx4:idx4 + i4].reshape(i4, 9)
    else:
        dat4 = np.zeros((1, dataset1.shape[1]), dtype=np.float32)
        bas4 = np.zeros((1, 9), dtype=np.float32)

    # pad to padding_size
    X1 = pad_to(dat1, padding_size, pad_value=0.0)
    X2 = pad_to(dat2, padding_size, pad_value=0.0)
    X3 = pad_to(dat3, padding_size, pad_value=0.0)
    X4 = pad_to(dat4, padding_size, pad_value=0.0)

    B1 = pad_to(bas1, padding_size, pad_value=0.0)
    B2 = pad_to(bas2, padding_size, pad_value=0.0)
    B3 = pad_to(bas3, padding_size, pad_value=0.0)
    B4 = pad_to(bas4, padding_size, pad_value=0.0)

    # 掩码：1 表示该位置有效，0 表示 pad
    C_m = np.zeros((padding_size,), dtype=np.float32)
    C_m[:i1] = 1.0
    H_m = np.zeros((padding_size,), dtype=np.float32)
    H_m[:i2] = 1.0
    N_m = np.zeros((padding_size,), dtype=np.float32)
    N_m[:i3] = 1.0
    O_m = np.zeros((padding_size,), dtype=np.float32)
    O_m[:i4] = 1.0

    # 重新 reshape 为 3D： (1, padding_size, feat_dim)
    X_3D1 = X1.reshape(1, padding_size, X1.shape[1])
    X_3D2 = X2.reshape(1, padding_size, X2.shape[1])
    X_3D3 = X3.reshape(1, padding_size, X3.shape[1])
    X_3D4 = X4.reshape(1, padding_size, X4.shape[1])

    basis1 = B1.reshape(1, padding_size, 9)
    basis2 = B2.reshape(1, padding_size, 9)
    basis3 = B3.reshape(1, padding_size, 9)
    basis4 = B4.reshape(1, padding_size, 9)

    C_m = C_m.reshape(1, padding_size, 1)
    H_m = H_m.reshape(1, padding_size, 1)
    N_m = N_m.reshape(1, padding_size, 1)
    O_m = O_m.reshape(1, padding_size, 1)

    return X_3D1, X_3D2, X_3D3, X_3D4, basis1, basis2, basis3, basis4, C_m, H_m, N_m, O_m


def dos_mask(C_m: np.ndarray, H_m: np.ndarray, N_m: np.ndarray, O_m: np.ndarray, padding_size: int):
    """
    把每个元素类别的掩码 (1, padding_size, 1) → (1, padding_size, 341)，
    其中 341 是 DOS 曲线的采样点数（固定值，可根据实际调整）。
    """
    def expand_mask(mask):
        # mask: shape (1, padding_size, 1) → reshape (1, padding_size) → repeat 341 次 → reshape (1, padding_size, 341)
        m = mask.reshape(1, padding_size)
        m_r = np.repeat(m, 341, axis=1)
        return m_r.reshape(1, padding_size, 341)

    C_d = expand_mask(C_m)
    H_d = expand_mask(H_m)
    N_d = expand_mask(N_m)
    O_d = expand_mask(O_m)
    return C_d, H_d, N_d, O_d


def get_efp_data(data_list: list):
    """
    从一批文件夹中，生成能量训练所需数据：
    返回：
      ener_ref         : list of energy references (垂直拼接后 shape=(n_samples, 1))
      forces_pre_list  : list of per-sample forces (每项是 (n_atoms, 3) numpy)
      press_ref        : list of pressures (垂直拼接后 shape=(n_samples, 1))
      X_pre_list       : list of per-sample原子指纹 (每项是 (n_atoms, feat_dim) numpy)
      basis_pre_list   : list of per-sample基函数 (每项是 (n_atoms, 9) numpy)
      X_at             : 原子数列表 (垂直拼接 shape=(n_samples, 1))
      X_el             : 电子数列表 (垂直拼接 shape=(n_samples, 1))
      X_at_elem        : per-sample per-element counts (垂直拼接 shape=(n_samples, 4))
    """
    ener_list = []
    forces_pre_list = []
    press_list = []
    X_pre_list = []
    basis_pre_list = []
    At_list = []
    El_list = []
    X_at_elem = []

    for folder in data_list:
        vol, supercell, dim, total_elec, elems_list, poscar_data = get_def_data(folder)
        El_list.append([total_elec])

        # 生成原子指纹和基函数
        dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(poscar_data, supercell, elems_list)
        At_list.append([num_atoms])
        X_pre_list.append(dset.copy())
        basis_pre_list.append(basis_mat.copy())
        X_at_elem.append(at_elem)

        # 获取 DFT 能量、力、压力
        Prop_e, forces_data, press = e_train(folder, num_atoms)
        ener_list.append([Prop_e])
        forces_pre_list.append(np.array(forces_data))
        press_list.append([press])

    X_elem = np.vstack(X_at_elem)           # shape = (n_samples, 4)
    X_at = np.vstack(At_list)               # shape = (n_samples, 1)
    X_el = np.vstack(El_list)               # shape = (n_samples, 1)
    press_ref = np.vstack(press_list)       # shape = (n_samples, 1)
    ener_ref = np.vstack(ener_list)         # shape = (n_samples, 1)
    return ener_ref, forces_pre_list, press_ref, X_pre_list, basis_pre_list, X_at, X_el, X_elem


def pad_dat(X_at_elem: np.ndarray, X_pre_list: list, padding_size: int):
    """
    将给定一组“原子指纹 (X_pre_list)” 和 “每个结构 per-element counts (X_at_elem)” pad 到统一大小：
    返回：
      X_1, X_2, X_3, X_4 : shape = (n_samples, padding_size, feat_dim)
      C_m, H_m, N_m, O_m : shape = (n_samples, padding_size, 1)
    """
    X_list = []
    C_list = []
    H_list = []
    N_list = []
    O_list = []

    for at_elem, dataset1 in zip(X_at_elem, X_pre_list):
        # 获取单结构 pad 后的指纹拼接
        X_pad = get_fp_all(at_elem, dataset1, padding_size)  # (padding_size, feat_dim*4)
        X_list.append(X_pad.T)  # 为了后续 reshape，先转置

        # 构造掩码
        C_at = np.zeros((padding_size,), dtype=np.float32); C_at[: at_elem[0]] = 1.0
        H_at = np.zeros((padding_size,), dtype=np.float32); H_at[: at_elem[1]] = 1.0
        N_at = np.zeros((padding_size,), dtype=np.float32); N_at[: at_elem[2]] = 1.0
        O_at = np.zeros((padding_size,), dtype=np.float32); O_at[: at_elem[3]] = 1.0

        C_list.append(C_at)
        H_list.append(H_at)
        N_list.append(N_at)
        O_list.append(O_at)

    X_stack = np.vstack(X_list)  # shape = (n_samples * 4, feat_dim)
    tot_conf = X_stack.shape[0] // (4 * padding_size)
    # 先 reshape 回 (n_samples, 4, padding_size, feat_dim_per_element)
    feat_per_elem = X_list[0].shape[0]  # 其实是 360，但我们只知道总列数
    X_3D = X_stack.reshape(tot_conf, 4, padding_size, feat_per_elem)

    X_1 = X_3D[:, 0, :, :]
    X_2 = X_3D[:, 1, :, :]
    X_3 = X_3D[:, 2, :, :]
    X_4 = X_3D[:, 3, :, :]

    C_m = np.vstack(C_list).reshape(tot_conf, padding_size, 1)
    H_m = np.vstack(H_list).reshape(tot_conf, padding_size, 1)
    N_m = np.vstack(N_list).reshape(tot_conf, padding_size, 1)
    O_m = np.vstack(O_list).reshape(tot_conf, padding_size, 1)

    return X_1, X_2, X_3, X_4, C_m, H_m, N_m, O_m


def pad_efp_data(
    X_at_elem: np.ndarray,
    X_pre_list: list,
    forces_pre_list: list,
    basis_pre_list: list,
    padding_size: int
):
    """
    将能量训练数据进行 pad：
    - X_pre_list: list of 原子指纹，per-sample shape=(n_atoms, feat_dim)
    - forces_pre_list: list of per-sample forces，per-sample shape=(n_atoms, 3)
    - basis_pre_list: list of per-sample basis_mat，per-sample shape=(n_atoms, 9)
    - X_at_elem: per-sample at_elem, shape=(n_samples, 4)

    返回:
      forces1~4, X_1~4, basis1~4, C_m~O_m
      其中 forces1~4: (n_samples, padding_size, 3)
            X_1~4:      (n_samples, padding_size, feat_dim)
            basis1~4:  (n_samples, padding_size, 9)
            C_m~O_m:   (n_samples, padding_size, 1)
    """
    X_list = []
    basis_list = []
    forces_list = []
    C_list = []
    H_list = []
    N_list = []
    O_list = []

    for at_elem, dataset1, forces_data, basis_mat in zip(X_at_elem, X_pre_list, forces_pre_list, basis_pre_list):
        X_pad, basis_pad, forces_pad = get_fp_basis_F(at_elem, dataset1, forces_data, basis_mat, padding_size)
        X_list.append(X_pad.T)        # shape = (padding_size * 4, ...)
        basis_list.append(basis_pad.T)   # shape = (padding_size * 4, ...)
        forces_list.append(forces_pad.T) # shape = (padding_size * 4, ...)

        # 构造掩码
        C_at = np.zeros((padding_size,), dtype=np.float32); C_at[: at_elem[0]] = 1.0
        H_at = np.zeros((padding_size,), dtype=np.float32); H_at[: at_elem[1]] = 1.0
        N_at = np.zeros((padding_size,), dtype=np.float32); N_at[: at_elem[2]] = 1.0
        O_at = np.zeros((padding_size,), dtype=np.float32); O_at[: at_elem[3]] = 1.0

        C_list.append(C_at)
        H_list.append(H_at)
        N_list.append(N_at)
        O_list.append(O_at)

    # Stack 成 (n_samples * 4 * padding_size, feat_dim)
    X_stack = np.vstack(X_list)
    basis_stack = np.vstack(basis_list)
    forces_stack = np.vstack(forces_list)

    tot_conf = X_stack.shape[0] // (4 * padding_size)
    feat_dim = X_list[0].shape[0]  # 每段的列数
    X_3D = X_stack.reshape(tot_conf, 4, padding_size, feat_dim)
    basis_3D = basis_stack.reshape(tot_conf, 4, padding_size, 9)
    forces_3D = forces_stack.reshape(tot_conf, 4, padding_size, 3)

    X_1 = X_3D[:, 0, :, :]
    X_2 = X_3D[:, 1, :, :]
    X_3 = X_3D[:, 2, :, :]
    X_4 = X_3D[:, 3, :, :]

    basis1 = basis_3D[:, 0, :, :]
    basis2 = basis_3D[:, 1, :, :]
    basis3 = basis_3D[:, 2, :, :]
    basis4 = basis_3D[:, 3, :, :]

    forces1 = forces_3D[:, 0, :, :]
    forces2 = forces_3D[:, 1, :, :]
    forces3 = forces_3D[:, 2, :, :]
    forces4 = forces_3D[:, 3, :, :]

    C_m = np.vstack(C_list).reshape(tot_conf, padding_size, 1)
    H_m = np.vstack(H_list).reshape(tot_conf, padding_size, 1)
    N_m = np.vstack(N_list).reshape(tot_conf, padding_size, 1)
    O_m = np.vstack(O_list).reshape(tot_conf, padding_size, 1)

    return forces1, forces2, forces3, forces4, X_1, X_2, X_3, X_4, basis1, basis2, basis3, basis4, C_m, H_m, N_m, O_m


def pad_dos_dat(Prop_vbcb: np.ndarray, X_1: np.ndarray, C_m: np.ndarray, H_m: np.ndarray, N_m: np.ndarray, O_m: np.ndarray, padding_size: int):
    """
    将 DOS 训练相关数据 pad：
    - Prop_vbcb: shape = (n_samples, 2) (VB, CB)
    - X_1:      shape = (n_samples, padding_size, feat_dim)
    - C_m~O_m: shape = (n_samples, padding_size, 1)
    返回：
      Prop_B:  (n_samples, 2)
      C_d ~ O_d: (n_samples, padding_size, 341)
    """
    tot_conf = X_1.shape[0]
    Prop_B = Prop_vbcb.reshape(tot_conf, 2)

    def expand_mask(mask):
        # mask: (n_samples, padding_size, 1) → reshape (n_samples, padding_size) → repeat 341 → reshape back
        m = mask.reshape(tot_conf, padding_size)
        m_r = np.repeat(m, 341, axis=1)
        return m_r.reshape(tot_conf, padding_size, 341)

    C_d = expand_mask(C_m)
    H_d = expand_mask(H_m)
    N_d = expand_mask(N_m)
    O_d = expand_mask(O_m)

    return Prop_B, C_d, H_d, N_d, O_d


def get_e_dos_data(data_list: list):
    """
    从一批文件夹中提取联合“能量 + DOS”训练数据：
    返回：
      ener_ref         : (n_samples, 1)
      forces_pre_list  : list of per-sample forces (n_atoms, 3)
      press_ref        : (n_samples, 1)
      X_pre_list       : list of per-sample 指纹 (n_atoms, feat_dim)
      basis_pre_list   : list of per-sample 基函数 (n_atoms, 9)
      X_at             : (n_samples, 1)
      X_el             : (n_samples, 1)
      X_at_elem        : (n_samples, 4)
      Prop_dos         : (n_samples, n_dos_points)  DOS 曲线
      Prop_vbcb        : (n_samples, 2)           (VB, CB)
    """
    Prop_dos_list = []
    Prop_vbcb_list = []
    ener_list = []
    forces_pre_list = []
    press_list = []
    X_pre_list = []
    basis_pre_list = []
    At_list = []
    El_list = []
    X_at_elem = []

    for folder in data_list:
        vol, supercell, dim, total_elec, elems_list, poscar_data = get_def_data(folder)
        El_list.append([total_elec])

        dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(poscar_data, supercell, elems_list)
        At_list.append([num_atoms])
        X_pre_list.append(dset.copy())
        basis_pre_list.append(basis_mat.copy())
        X_at_elem.append(at_elem)

        Prop_e, forces_data, press = e_train(folder, num_atoms)
        dos_dat, VB, CB = dos_data(folder, total_elec)
        ener_list.append([Prop_e])
        forces_pre_list.append(np.array(forces_data))
        press_list.append([press])

        Prop_dos_list.append(dos_dat.copy())
        Prop_vbcb_list.append([VB, CB])

    X_elem = np.vstack(X_at_elem)
    X_at = np.vstack(At_list)
    X_el = np.vstack(El_list)
    Prop_dos = np.vstack(Prop_dos_list)
    Prop_vbcb = np.vstack(Prop_vbcb_list)
    press_ref = np.vstack(press_list)
    ener_ref = np.vstack(ener_list)

    return ener_ref, forces_pre_list, press_ref, X_pre_list, basis_pre_list, X_at, X_el, X_elem, Prop_dos, Prop_vbcb


def get_dos_data(data_list: list):
    """
    从一批文件夹中提取 DOS 训练数据：
    返回：
      X_pre_list    : list of per-sample 指纹 (n_atoms, feat_dim)
      X_at          : (n_samples, 1)
      X_el          : (n_samples, 1)
      X_at_elem     : (n_samples, 4)
      Prop_dos      : (n_samples, n_dos_points)  DOS 曲线
      Prop_vbcb     : (n_samples, 2)            (VB, CB)
    """
    Prop_dos_list = []
    Prop_vbcb_list = []
    X_pre_list = []
    At_list = []
    El_list = []
    X_at_elem = []

    for folder in data_list:
        vol, supercell, dim, total_elec, elems_list, poscar_data = get_def_data(folder)
        El_list.append([total_elec])

        dset, basis_mat, sites_elem, num_atoms, at_elem = fp_atom(poscar_data, supercell, elems_list)
        At_list.append([num_atoms])
        X_pre_list.append(dset.copy())
        X_at_elem.append(at_elem)

        dos_dat, VB, CB = dos_data(folder, total_elec)
        Prop_dos_list.append(dos_dat.copy())
        Prop_vbcb_list.append([VB, CB])

    X_elem = np.vstack(X_at_elem)
    X_at = np.vstack(At_list)
    X_el = np.vstack(El_list)
    Prop_dos = np.vstack(Prop_dos_list)
    Prop_vbcb = np.vstack(Prop_vbcb_list)

    return X_pre_list, X_at, X_el, X_elem, Prop_dos, Prop_vbcb


def get_dos_e_train_data(X_1: np.ndarray, X_2: np.ndarray, X_3: np.ndarray, X_4: np.ndarray,
                         X_elem: np.ndarray, padding_size: int, modelCHG):
    """
    对一批经 pad 后的指纹 (X_1~X_4) 做电荷加权，并返回用于能量/DOS 训练的加权指纹：
      - X_1~X_4: shape = (n_samples, padding_size, feat_dim)
      - X_elem:  shape = (n_samples, 4)
      - modelCHG: 已加载的电荷模型，用于计算每个原子的电荷系数

    返回:
      X_C, X_H, X_N, X_O: shape = (n_samples, padding_size, feat_dim+1)
    """
    n_samples = X_1.shape[0]
    X_C_list, X_H_list, X_N_list, X_O_list = [], [], [], []

    for i in range(n_samples):
        # 先取出单个样本的各元素 pad 后指纹： (padding_size, feat_dim)
        x1 = X_1[i].reshape(1, padding_size, -1)
        x2 = X_2[i].reshape(1, padding_size, -1)
        x3 = X_3[i].reshape(1, padding_size, -1)
        x4 = X_4[i].reshape(1, padding_size, -1)

        # 电荷系数预测（coef_predict 仍然调用原始 Keras/TensorFlow 实现）
        coef1, coef2, coef3, coef4 = coef_predict(
            x1, x2, x3, x4,
            X_elem[i][0], X_elem[i][1], X_elem[i][2], X_elem[i][3],
            modelCHG
        )
        # coef# 的形状： (1, padding_size, 1)

        # 将 coef 转置到 (padding_size, 1)
        c1 = coef1.reshape(padding_size, 1)
        c2 = coef2.reshape(padding_size, 1)
        c3 = coef3.reshape(padding_size, 1)
        c4 = coef4.reshape(padding_size, 1)

        # 拼接：指纹 + 电荷系数 → (padding_size, feat_dim + 1)
        X_C_list.append(np.concatenate([X_1[i], c1], axis=1))
        X_H_list.append(np.concatenate([X_2[i], c2], axis=1))
        X_N_list.append(np.concatenate([X_3[i], c3], axis=1))
        X_O_list.append(np.concatenate([X_4[i], c4], axis=1))

    X_C = np.stack(X_C_list, axis=0)
    X_H = np.stack(X_H_list, axis=0)
    X_N = np.stack(X_N_list, axis=0)
    X_O = np.stack(X_O_list, axis=0)
    return X_C, X_H, X_N, X_O
