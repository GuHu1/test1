"""LiDAR 点云处理与相机深度图生成/补全。

合并原 OSZMiner 的 ``dataset/lidar.py`` 与
``dataset/depth_completion.py``：

- ``aggregate_lidar_sweeps``：把当前与历史扫描（含自车运动补偿）
  聚合为单一 ego 系点云，是深度投影前的点源。
- ``filter_ground_points``：剔除地面点，消除路面幻影障碍。
- ``_get_transform`` / ``_ego_pose_tf``：从 nuScenes 记录取
  sensor -> ego / ego -> global 变换矩阵。
- ``project_lidar_to_camera``：ego 系点 -> 逐像素稀疏深度。
- ``densify_depth_map``：边界感知的近邻插值补全。

兼容性：Python 3.8 / numpy 1.19.5 / scipy 1.10.1 /
nuscenes-devkit 1.1.9。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scipy.spatial import cKDTree

from .osz_config import Z_MIN_M

try:
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("[WARN] 未找到 nuscenes-devkit，改用合成 mock 数据。")


# ---------------------------------------------------------------------------
# LiDAR 扫描聚合、地面过滤与传感器位姿工具
# ---------------------------------------------------------------------------
def _get_transform(nusc, record) -> np.ndarray:
    """返回标定传感器 -> ego 的变换矩阵（4x4 float32）。"""
    cs = nusc.get('calibrated_sensor', record['calibrated_sensor_token'])
    T = transform_matrix(
        cs['translation'],
        Quaternion(cs['rotation']),
        inverse=False,
    )
    return T.astype(np.float32)


def filter_ground_points(pts_ego: np.ndarray,
                         z_thresh: float = Z_MIN_M) -> np.ndarray:
    """剔除地面及以下的 LiDAR 点。

    地面点投影到相机深度图会产生虚假障碍体：15 m 外的地面与 15 m
    外 0.4 m 高的障碍物深度相同。投影前先过滤，可消除路面上幻影
    OSZ 的最大来源。

    Parameters
    ----------
    pts_ego : np.ndarray
        ego 系 LiDAR 点，形状 ``(N, 3)``，``x`` 前向、``y`` 左向、
        ``z`` 上向。
    z_thresh : float, optional
        ``z_ego < z_thresh`` 的点判为地面并剔除。默认 ``Z_MIN_M``。

    Returns
    -------
    np.ndarray
        过滤后的点，形状 ``(M, 3)``，仅保留 ``z_ego >= z_thresh``。
    """
    return pts_ego[pts_ego[:, 2] >= z_thresh]


def _ego_pose_tf(nusc, sample_token: str):
    """由 LiDAR sample_data 的 ego pose 构建 ego -> global 变换。"""
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    T = transform_matrix(ep['translation'], Quaternion(ep['rotation']), inverse=False)
    return T.astype(np.float64)


def aggregate_lidar_sweeps(nusc, sample_token: str,
                           n_sweeps: int = 0,
                           ground_z_thresh: float = Z_MIN_M) -> np.ndarray:
    """将当前与历史 LiDAR 扫描聚合为一个 ego 系点云。

    加载关键帧 LiDAR 扫描及最多 ``n_sweeps`` 个历史扫描（沿
    ``sample_data['prev']`` 链，仅取过去）。历史扫描的点先做自车
    运动补偿（past_ego -> global -> current_ego），再经地面过滤后
    拼接。

    该函数完全绕开深度图管线：调用方可以直接对返回点云做 3D 体素
    化，避免补全或投射误差。

    Parameters
    ----------
    nusc : nuscenes.nuscenes.NuScenes
        已初始化的 nuScenes 数据集实例。
    sample_token : str
        当前帧的关键帧样本 token。
    n_sweeps : int, optional
        纳入的历史扫描数。``0`` 表示仅关键帧。默认 0。
    ground_z_thresh : float, optional
        地面剔除的 ``z_ego`` 截止高度（米）。默认 ``Z_MIN_M``。

    Returns
    -------
    np.ndarray
        当前 ego 系下的全部聚合点，形状 ``(M, 3)``，dtype
        ``float32``。地面点已剔除。
    """
    sample = nusc.get('sample', sample_token)
    lidar_tok = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_tok)

    # 当前 ego -> global
    T_cur_ego2global = _ego_pose_tf(nusc, sample_token)

    all_pts = []

    # 当前关键帧扫描
    pc = LidarPointCloud.from_file(str(Path(nusc.dataroot) / lidar_sd['filename']))
    T_l2e = _get_transform(nusc, lidar_sd)
    pts_h = np.concatenate([pc.points[:3].T, np.ones((pc.points.shape[1], 1))], axis=1)
    pts_ego = (T_l2e @ pts_h.T).T[:, :3]
    pts_ego = filter_ground_points(pts_ego, ground_z_thresh)
    all_pts.append(pts_ego.astype(np.float32))

    # 历史扫描（prev 链，仅过去）
    tok = lidar_sd['prev']
    for _ in range(n_sweeps):
        if not tok:
            break
        sd = nusc.get('sample_data', tok)

        # 读取历史扫描点 -> 历史 ego 系
        pc_p = LidarPointCloud.from_file(str(Path(nusc.dataroot) / sd['filename']))
        T_l2e_p = _get_transform(nusc, sd)
        pts_p_h = np.concatenate([pc_p.points[:3].T,
                                  np.ones((pc_p.points.shape[1], 1))], axis=1)
        pts_p_ego = (T_l2e_p @ pts_p_h.T).T[:, :3]

        # 运动补偿前先做地面过滤（z 位于历史 ego 系）
        pts_p_ego = filter_ground_points(pts_p_ego, ground_z_thresh)
        if len(pts_p_ego) == 0:
            tok = sd['prev']
            continue

        # 自车运动补偿：past_ego -> global -> current_ego
        ep = nusc.get('ego_pose', sd['ego_pose_token'])
        T_past_ego2global = transform_matrix(
            ep['translation'], Quaternion(ep['rotation']), inverse=False
        ).astype(np.float64)
        pts_global_h = (T_past_ego2global @
                        np.concatenate([pts_p_ego,
                                        np.ones((len(pts_p_ego), 1))], axis=1).T).T
        pts_cur_ego_h = (np.linalg.inv(T_cur_ego2global) @ pts_global_h.T).T
        pts_cur_ego = pts_cur_ego_h[:, :3].astype(np.float32)

        all_pts.append(pts_cur_ego)

        tok = sd['prev']

    print(f"  [aggregate] {len(all_pts)} sweeps, "
          f"{sum(len(p) for p in all_pts)} points total")
    return np.concatenate(all_pts, axis=0)


# ---------------------------------------------------------------------------
# 相机深度图的生成与补全
# ---------------------------------------------------------------------------
def densify_depth_map(depth_map: np.ndarray,
                      max_radius: int = 16,
                      depth_discontinuity_thresh: float = 4.0) -> np.ndarray:
    """近邻插值补全稀疏深度图。

    朴素最近邻插值会涂抹物体边界：车辆旁的背景空洞可能被车辆表面
    的深度填充（其最近的有效像素落在车面上）。这会加宽车辆轮廓，
    反投影后膨胀遮挡体，使 OSZ 超出真实车宽。

    因此对每个无效像素检查最近的 4 个有效邻居：若它们的深度跨度
    超过 ``depth_discontinuity_thresh``，说明该像素位于深度边缘，
    保持未知（0）；否则取最近邻居的深度。

    Parameters
    ----------
    depth_map : np.ndarray
        稀疏深度图，形状 ``(H, W)``。0 表示无测量。
    max_radius : int, optional
        填充无效像素的最大搜索半径（像素）。距任何有效测量都超过
        该半径的像素保持 0。默认 16。
    depth_discontinuity_thresh : float, optional
        深度跨度阈值（米）。最近 4 个有效邻居的深度范围超过该值时，
        该像素被视为深度不连续处，不填充。默认 4.0。

    Returns
    -------
    np.ndarray
        补全后的深度图，形状 ``(H, W)``，dtype ``float32``。
        原有效像素保持不变；未填充像素仍为 0。

    Notes
    -----
    默认参数按误差审计结论调优：``max_radius`` 从 8 提到 16 以填满
    较大的物体内部；``depth_discontinuity_thresh`` 从 1.5 提到 4.0
    以减少导致 BEV 占据碎片化的边缘空洞。
    """
    H, W = depth_map.shape
    valid = depth_map > 0
    if valid.sum() == 0:
        return depth_map.copy()

    coords = np.array(np.nonzero(valid)).T          # (N, 2)  (y, x)
    values = depth_map[valid]

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid_coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)

    tree = cKDTree(coords)

    # 查询 K=4 近邻用于检测深度不连续边界
    k_neighbors = min(4, len(coords))
    dist_k, idx_k = tree.query(grid_coords, k=k_neighbors)
    if k_neighbors == 1:
        dist_k = dist_k[:, None]
        idx_k = idx_k[:, None]

    values_k = values[idx_k]                          # (H*W, k) 候选深度
    depth_spread = values_k.max(axis=1) - values_k.min(axis=1)  # (H*W,)

    # 默认插值取 k=1 最近邻的深度与距离
    nearest_dist = dist_k[:, 0]
    nearest_depth = values_k[:, 0]

    # 若最近的有效邻居深度跨度大（例如像素位于车辆边缘与背景之间），
    # 则该像素处于深度不连续处，插值不可靠，保持未知（0）。
    is_discontinuous = depth_spread > depth_discontinuity_thresh

    dense = nearest_depth.copy()
    dense[is_discontinuous] = 0.0
    dense = dense.reshape(H, W)
    dist_map = nearest_dist.reshape(H, W)

    # 不填充远离任何有效测量的像素
    dense_mask = dist_map <= max_radius
    dense = dense * dense_mask.astype(np.float32)

    # 原有效像素保持不变
    dense[valid] = depth_map[valid]
    return dense


def project_lidar_to_camera(
    points_ego: np.ndarray,
    K: np.ndarray,
    T_cam2ego: np.ndarray,
    img_h: int,
    img_w: int,
    min_dist: float = 1.0,
) -> np.ndarray:
    """将 ego 系 LiDAR 点投影到相机像平面。

    Parameters
    ----------
    points_ego : np.ndarray
        ego 系 LiDAR 点，形状 ``(N, 3)``。
    K : np.ndarray
        相机内参矩阵，形状 ``(3, 3)``。
    T_cam2ego : np.ndarray
        相机 -> ego 刚体变换，形状 ``(4, 4)``。
    img_h : int
        输出图像高（像素）。
    img_w : int
        输出图像宽（像素）。
    min_dist : float, optional
        保留的最小相机系深度；``z_cam <= min_dist`` 的点被丢弃。
        默认 1.0。

    Returns
    -------
    np.ndarray
        深度图，形状 ``(img_h, img_w)``，dtype ``float32``。无测量的
        像素为 0。多点落在同一像素时保留最近的点。
    """
    T_ego2cam = np.linalg.inv(T_cam2ego)

    # 变换到相机系
    pts_h = np.concatenate([points_ego, np.ones((len(points_ego), 1))], axis=1)
    pts_cam = (T_ego2cam @ pts_h.T).T[:, :3]  # (N, 3)

    # 只保留相机前方的点
    mask = pts_cam[:, 2] > min_dist
    pts_cam = pts_cam[mask]

    # 投影
    uvw = (K @ pts_cam.T).T  # (M, 3)
    z = uvw[:, 2]
    u = (uvw[:, 0] / z).astype(np.int32)
    v = (uvw[:, 1] / z).astype(np.int32)

    # 过滤到图像边界内
    in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, z = u[in_img], v[in_img], z[in_img]

    # 构建深度图（每像素保留最近的点）
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    # 按深度降序排序，使近处的点最后写入并覆盖远处的点
    order = np.argsort(-z)
    depth_map[v[order], u[order]] = z[order]
    return depth_map
