# dftpy/fp.py

import numpy as np
from pymatgen.io.vasp.outputs import Poscar
import itertools

# Constants for DOS mask (fixed number of energy sampling points)
DOS_NUM_POINTS = 341

# ------------------------------------------------------------------------------
# Helper function: pad a 2D array (n_rows × n_feats) up to (target_rows × n_feats)
# ------------------------------------------------------------------------------
def pad_to(arr: np.ndarray, target_rows: int, pad_value=0.0):
    """
    将二维数组 arr 沿第 0 轴 pad 到 (target_rows, n_feats)。
    如果 arr.shape[0] >= target_rows，直接返回 arr.copy()；否则在底部用 pad_value 填充。
    """
    n_rows, n_feats = arr.shape
    if n_rows >= target_rows:
        return arr.copy()
    pad_amount = target_rows - n_rows
    pad_block = np.full((pad_amount, n_feats), pad_value, dtype=arr.dtype)
    return np.vstack([arr, pad_block])


# ------------------------------------------------------------------------------
# Core fingerprint function
# ------------------------------------------------------------------------------
def fp_atom(
    structure,
    grid_spacing: float,
    cut_off_rad: float,
    widest_gaussian: float,
    narrowest_gaussian: float,
    num_gamma: int
):
    """
    生成“原子指纹”及相关几何信息，替代原 fp.py 中基于 TensorFlow/Keras 的实现。
    本函数计算三类特征：
      1. 径向(radial) fingerprint
      2. 方向性(dipole) fingerprint
      3. 二阶(quadrupole) fingerprint

    输入：
      - structure       : pymatgen.Structure 对象（对应 POSCAR 文件）
      - grid_spacing    : 网格间隔 (float)，暂不直接使用，可用于后续扩展
      - cut_off_rad     : 截断半径 (float)
      - widest_gaussian : 最宽 Gaussian 宽度 (float)
      - narrowest_gaussian: 最窄 Gaussian 宽度 (float)
      - num_gamma       : 使用的 Gaussian 数量 (int)

    输出：
      - dset            : np.ndarray, shape=(n_atoms, feature_dim)
                          每一行是一个原子的指纹向量 (radial + dipole + quadrupole)
      - basis_mat       : np.ndarray, shape=(n_atoms, 9)
                          每个原子的局部坐标框架（正交矩阵 flatten 后的 9 元素）
      - sites_elem      : list of Site 对象（pymatgen），分元素顺序排列
      - num_atoms       : int，总原子数
      - at_elem         : list of int，[n_C, n_H, n_N, n_O] 表示每种元素的原子数量
    """
    # 1. 根据 structure 提取元素顺序和每种元素对应的子结构
    elems = sorted(set(structure.site_symbols))
    # 要求 elems 顺序为 ['C', 'H', 'N', 'O'] 中的子集。缺少的元素顺序后补 0。
    # 最终 at_elem 长度固定为 4，对应 C, H, N, O
    elem_to_index = {'C': 0, 'H': 1, 'N': 2, 'O': 3}
    at_elem = [0, 0, 0, 0]
    sites_elem = [[], [], [], []]

    # 收集每个元素类别的原子列表
    for site in structure.sites:
        sym = site.specie.symbol
        if sym in elem_to_index:
            idx = elem_to_index[sym]
            at_elem[idx] += 1
            sites_elem[idx].append(site)
        else:
            # 如果遇到不是 C/H/N/O 的元素，则忽略（或根据需要自行扩展）
            pass

    num_atoms = structure.num_sites

    # 2. 构建所有元素原子的笛卡尔坐标列表（cart_grid）和唯一邻居位置列表（unique_cart_list）
    cart_grid_list = []
    unique_cart_list = []
    for idx, elem in enumerate(['C', 'H', 'N', 'O']):
        if at_elem[idx] == 0:
            continue
        # 从原始结构中删除其他元素，只保留当前元素
        sub = structure.copy()
        remove_list = [e for e in elems if e != elem]
        sub.remove_species(remove_list)

        # 当前元素的所有原子坐标（笛卡尔）
        coords_cart = np.array(sub.cart_coords)
        cart_grid_list.append(coords_cart)

        # 对每个该元素原子，找到其邻居的分数坐标，再统一成唯一列表
        all_frac = []
        for site in sub.sites:
            neighs = structure.get_neighbors(site, cut_off_rad)
            for neigh in neighs:
                all_frac.append(neigh[0].frac_coords)
            # 自己的分数坐标也加入
            all_frac.append(site.frac_coords)

        if len(all_frac) == 0:
            frac_array_unique = np.array(sub.frac_coords)
        else:
            frac_arr = np.array(all_frac)
            # 找出唯一的分数坐标
            uniq = np.unique(frac_arr.view([('', frac_arr.dtype)] * frac_arr.shape[1]))
            frac_array_unique = uniq.view(frac_arr.dtype).reshape((uniq.shape[0], frac_arr.shape[1]))

        cart_unique = np.dot(frac_array_unique, sub.lattice.matrix)
        unique_cart_list.append(cart_unique)

    # 把所有元素原子的笛卡尔网格点拼接成一个大数组
    if len(cart_grid_list) == 0:
        # 如果结构不含 C/H/N/O 中的任何元素，则返回空特征
        return np.zeros((0, 0)), np.zeros((0, 9)), [], 0, [0, 0, 0, 0]

    cart_grid_K = np.vstack(cart_grid_list)  # shape = (num_atoms, 3)
    padding_size = max(at_elem)

    # 3. 生成 gamma 列表（与原始 TensorFlow 代码一致）
    sigma_list = np.logspace(np.log10(narrowest_gaussian), np.log10(widest_gaussian), num_gamma)
    gamma_list = 0.5 / (sigma_list**2)

    # 4. 计算距离矩阵 rad_diff, rad, rad_inv, 和截断函数 cut_off_func
    #    rad_diff: shape = (n_unique_neighbors, n_atoms, 3)
    #    这里我们先把 cart_grid_K 作为当前原子位置集合，unique_cart_list 分元素顺序存放，
    #    所以我们需要对每个元素类别分别计算
    #
    # 为了尽量保持与原始逻辑一致，这里逐元素类别循环：
    all_radial = []
    all_dipole = []
    all_quad = []
    for elem_idx, cart_unique in enumerate(unique_cart_list):
        # cart_unique: shape = (n_neighbors_elem, 3)
        # cart_grid_K: shape = (num_atoms, 3)
        # rad_diff_elem: shape = (n_neighbors_elem, num_atoms, 3)
        rad_diff = cart_grid_K[None, :, :] - cart_unique[:, None, :]  # (n_unique, n_atoms, 3)
        # 计算每对的欧氏距离 rad: shape = (n_unique, n_atoms)
        rad = np.linalg.norm(rad_diff, axis=2)
        # rad_inv: 1/rad with保留分母为 0 的处理
        with np.errstate(divide='ignore', invalid='ignore'):
            rad_inv = np.where(rad != 0, 1.0 / rad, 0.0)

        # 截断函数 f_c(r) = (cos(min(r, R)/R * π) + 1)/2
        r_cut = np.minimum(rad, cut_off_rad)
        cut_off = (np.cos((r_cut / cut_off_rad) * np.pi) + 1.0) * 0.5  # shape = (n_unique, n_atoms)

        # 对每个 gamma 计算径向基函数和其加权叠加→radial_fp, shape=(num_gamma, num_atoms)
        rad_fp_elem = []
        dipole_elem = [[] for _ in range(3)]   # 分量 x,y,z 每个存 num_gamma×n_atoms
        quad_elem = [[] for _ in range(6)]     # 6 个二阶矩项

        for gamma in gamma_list:
            norm = (gamma / np.pi) ** 1.5  # 归一化常数
            gaussian = norm * np.exp(-gamma * (rad**2))  # shape=(n_unique, n_atoms)
            gaussian *= cut_off  # 截断

            # 1) 径向累加
            radial_fp_elem.append(np.sum(gaussian, axis=0))  # (n_atoms,)

            # 2) 偶极项：∑ ( (r_i - r_j) * gaussian / r )
            #    rad_diff[:,:,i] * gaussian / rad 进行分量累加
            for dim_i in range(3):
                num = rad_diff[:, :, dim_i] * gaussian
                dip_i = np.sum(np.where(rad != 0, num / rad, 0.0), axis=0)  # shape=(n_atoms,)
                dipole_elem[dim_i].append(dip_i)

            # 3) 四极张量项：需要 6 个组合
            #    r_x*r_y, r_y*r_z, r_z*r_x, r_x^2, r_y^2, r_z^2 加权累加
            rx = rad_diff[:, :, 0]
            ry = rad_diff[:, :, 1]
            rz = rad_diff[:, :, 2]
            # (r_x*r_x * gaussian / r^2), etc.，分母为 0 时取 0
            with np.errstate(divide='ignore', invalid='ignore'):
                q_xx = np.sum(np.where(rad != 0, (rx * rx * gaussian) / (rad**2), 0.0), axis=0)
                q_yy = np.sum(np.where(rad != 0, (ry * ry * gaussian) / (rad**2), 0.0), axis=0)
                q_zz = np.sum(np.where(rad != 0, (rz * rz * gaussian) / (rad**2), 0.0), axis=0)
                q_xy = np.sum(np.where(rad != 0, (rx * ry * gaussian) / (rad**2), 0.0), axis=0)
                q_yz = np.sum(np.where(rad != 0, (ry * rz * gaussian) / (rad**2), 0.0), axis=0)
                q_zx = np.sum(np.where(rad != 0, (rz * rx * gaussian) / (rad**2), 0.0), axis=0)

            quad_elem[0].append(q_xx)  # XX
            quad_elem[1].append(q_yy)  # YY
            quad_elem[2].append(q_zz)  # ZZ
            quad_elem[3].append(q_xy)  # XY
            quad_elem[4].append(q_yz)  # YZ
            quad_elem[5].append(q_zx)  # ZX

        # 将这个元素类别的 radial/dipole/quad 特征拼好
        # radial_fp_elem: list of num_gamma arrays, 每个 (n_atoms,)
        rad_fp_elem = np.stack(radial_fp_elem, axis=0)  # shape=(num_gamma, n_atoms)
        dipole_elem = np.stack([np.stack(dim_list, axis=0) for dim_list in dipole_elem], axis=0)
        # dipole_elem: shape=(3, num_gamma, n_atoms)  → reshape 成 (3*num_gamma, n_atoms)
        dipole_elem = dipole_elem.reshape(3 * num_gamma, num_atoms)
        # quad_elem: list of 6 lists，每个内层是长度 num_gamma 的 (n_atoms,) → (6, num_gamma, n_atoms)
        quad_elem = np.stack([np.stack(q_list, axis=0) for q_list in quad_elem], axis=0)
        # quad_elem: shape=(6, num_gamma, n_atoms) → reshape 成 (6*num_gamma, n_atoms)
        quad_elem = quad_elem.reshape(6 * num_gamma, num_atoms)

        # 最终把 radial/dipole/quad 沿行拼接 (num_gamma + 3*num_gamma + 6*num_gamma, n_atoms)
        all_elem_feat = np.concatenate([rad_fp_elem, dipole_elem, quad_elem], axis=0)
        # 转置成 (n_atoms, feature_dim_elem)
        feat_elem_T = all_elem_feat.T  # shape = (n_atoms, num_gamma * 10)
        all_radial.append(feat_elem_T)

    # 不同元素类别的特征沿列拼接
    # 如果只有 1 种元素，那么 all_radial 只有一个 (n_atoms, feat_dim_elem)，直接用它
    # 如果有多种，则 np.concatenate([...], axis=1)
    dset = np.concatenate(all_radial, axis=1)  # shape = (num_atoms, feature_dim_total)

    # 5. 构建基函数矩阵 basis_mat：对每个原子构造局部坐标系（3×3 正交矩阵），flatten 成 9 维
    #    这里复刻原脚本中寻找最近两个邻居然后构建正交基的思路
    basis_list = []
    cutoff_distance = 5.0
    # sites_elem 是一个长度 4 的 list，每个元素是该类别的 Site 列表
    for elem_idx in range(4):
        for site in sites_elem[elem_idx]:
            pos = site.coords
            # 找与该原子距离 < cutoff_distance 的所有邻居
            neighs = structure.get_neighbors(site, cutoff_distance)
            # 按距离排序
            neighs_sorted = sorted(neighs, key=lambda x: x[1])
            # 取最近两个邻居方向，构造局部坐标
            if len(neighs_sorted) < 2:
                # 如果邻居太少，使用单位矩阵
                mat = np.eye(3)
            else:
                v1 = neighs_sorted[0][0].coords - pos
                v2 = neighs_sorted[1][0].coords - pos
                u3 = np.cross(v1, v2)
                u2 = np.cross(v1, u3)
                u1 = v1 / np.linalg.norm(v1)
                u2 = u2 / np.linalg.norm(u2) if np.linalg.norm(u2) != 0 else np.array([1.0, 0.0, 0.0])
                u3 = u3 / np.linalg.norm(u3) if np.linalg.norm(u3) != 0 else np.array([0.0, 1.0, 0.0])
                mat = np.vstack((u1, u2, u3)).T  # shape=(3,3)

            basis_list.append(mat.flatten())  # flatten → 9

    # 如果 at_elem 中某个元素类别为 0，我们没有为它添加任何 basis，需补 0
    total_mat = np.vstack(basis_list)  # shape = (num_atoms, 9)

    # 6. 返回结果
    #    - dset:     shape = (num_atoms, feature_dim_total)
    #    - basis_mat:shape = (num_atoms, 9)
    #    - sites_elem:按元素分开后的 Site 列表（长度固定 4，空类别对应 []）
    #    - num_atoms: int
    #    - at_elem:  [n_C, n_H, n_N, n_O]
    return dset.astype(np.float32), total_mat.astype(np.float32), sites_elem, num_atoms, at_elem


# ------------------------------------------------------------------------------
# 带电荷归一化的指纹（fp_chg_norm）
# ------------------------------------------------------------------------------
def fp_chg_norm(
    Coef_at1: np.ndarray,
    Coef_at2: np.ndarray,
    Coef_at3: np.ndarray,
    Coef_at4: np.ndarray,
    X_3D1: np.ndarray,
    X_3D2: np.ndarray,
    X_3D3: np.ndarray,
    X_3D4: np.ndarray,
    padding_size: int,
    scaler_paths: tuple
):
    """
    将预测得到的电荷系数 Coef_ati 与对应元素的指纹 X_3Di 进行拼接，并做 MaxAbsScaler 归一化。

    输入：
      - Coef_at1~4: 形状 (1, padding_size, 1) 的电荷系数张量（NumPy）
      - X_3D1~4:    形状 (1, padding_size, feat_dim) 的指纹张量（NumPy）
      - padding_size: int
      - scaler_paths: (pathC, pathH, pathN, pathO)，分别指向训练好的 joblib scaler 文件

    输出：
      - X_C, X_H, X_N, X_O:    shape = (1, padding_size, feat_dim + 1)，加了电荷维
    """
    # 1. 将 Coef_at# 转置到 (padding_size, 1)
    c1 = Coef_at1.reshape(padding_size, 1)
    c2 = Coef_at2.reshape(padding_size, 1)
    c3 = Coef_at3.reshape(padding_size, 1)
    c4 = Coef_at4.reshape(padding_size, 1)

    # 2. 将 X_3Di reshape 到 (padding_size, feat_dim)
    x1 = X_3D1.reshape(padding_size, X_3D1.shape[-1])
    x2 = X_3D2.reshape(padding_size, X_3D2.shape[-1])
    x3 = X_3D3.reshape(padding_size, X_3D3.shape[-1])
    x4 = X_3D4.reshape(padding_size, X_3D4.shape[-1])

    # 3. 拼接电荷维度 (padding_size, feat_dim+1)
    X_C = np.concatenate([x1, c1], axis=1)
    X_H = np.concatenate([x2, c2], axis=1)
    X_N = np.concatenate([x3, c3], axis=1)
    X_O = np.concatenate([x4, c4], axis=1)

    # 4. 依次加载 sklearn scaler 对应文件，对每个 (padding_size, feat_dim+1) 做 MaxAbsScaler 变换
    from joblib import load
    scalerC = load(scaler_paths[0])
    scalerH = load(scaler_paths[1])
    scalerN = load(scaler_paths[2])
    scalerO = load(scaler_paths[3])

    X_C_flat = X_C.copy()
    X_H_flat = X_H.copy()
    X_N_flat = X_N.copy()
    X_O_flat = X_O.copy()

    X_C_scaled = scalerC.transform(X_C_flat)
    X_H_scaled = scalerH.transform(X_H_flat)
    X_N_scaled = scalerN.transform(X_N_flat)
    X_O_scaled = scalerO.transform(X_O_flat)

    # 5. reshape 回 (1, padding_size, feat_dim+1)
    featC = X_C_scaled.reshape(1, padding_size, X_C_scaled.shape[-1])
    featH = X_H_scaled.reshape(1, padding_size, X_H_scaled.shape[-1])
    featN = X_N_scaled.reshape(1, padding_size, X_N_scaled.shape[-1])
    featO = X_O_scaled.reshape(1, padding_size, X_O_scaled.shape[-1])

    return featC.astype(np.float32), featH.astype(np.float32), featN.astype(np.float32), featO.astype(np.float32)


# ------------------------------------------------------------------------------
# 仅指纹归一化 (fp_norm)
# ------------------------------------------------------------------------------
def fp_norm(
    X_C: np.ndarray,
    X_H: np.ndarray,
    X_N: np.ndarray,
    X_O: np.ndarray,
    padding_size: int,
    scaler_paths: tuple
):
    """
    对已经整批 pad 后的指纹张量进行 MaxAbsScaler 归一化（不再拼接新的电荷维度）。
    本函数假设 X_C/H/N/O 形状均为 (n_samples, padding_size, feat_dim)。

    输入：
      - X_C, X_H, X_N, X_O: 每个 shape = (n_samples, padding_size, feat_dim)
      - padding_size:       int
      - scaler_paths:       同上

    输出：
      - X_Cn, X_Hn, X_Nn, X_On: 归一化后同维度输出
    """
    from joblib import load
    scalerC = load(scaler_paths[0])
    scalerH = load(scaler_paths[1])
    scalerN = load(scaler_paths[2])
    scalerO = load(scaler_paths[3])

    n_samples = X_C.shape[0]
    feat_dim = X_C.shape[-1]

    # 先 reshape 到 (n_samples * padding_size, feat_dim)
    X_C_flat = X_C.reshape(n_samples * padding_size, feat_dim)
    X_H_flat = X_H.reshape(n_samples * padding_size, feat_dim)
    X_N_flat = X_N.reshape(n_samples * padding_size, feat_dim)
    X_O_flat = X_O.reshape(n_samples * padding_size, feat_dim)

    # 归一化
    X_C_scaled = scalerC.transform(X_C_flat)
    X_H_scaled = scalerH.transform(X_H_flat)
    X_N_scaled = scalerN.transform(X_N_flat)
    X_O_scaled = scalerO.transform(X_O_flat)

    # reshape 回 (n_samples, padding_size, feat_dim)
    X_Cn = X_C_scaled.reshape(n_samples, padding_size, feat_dim)
    X_Hn = X_H_scaled.reshape(n_samples, padding_size, feat_dim)
    X_Nn = X_N_scaled.reshape(n_samples, padding_size, feat_dim)
    X_On = X_O_scaled.reshape(n_samples, padding_size, feat_dim)

    return (
        X_Cn.astype(np.float32),
        X_Hn.astype(np.float32),
        X_Nn.astype(np.float32),
        X_On.astype(np.float32),
    )
