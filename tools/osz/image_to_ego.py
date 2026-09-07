"""将相机深度图反投影为 ego 系 3D 点。

给定单相机度量深度图、相机内参 ``K`` 与相机->ego 外参
``T_cam2ego``，本模块产出自车坐标系（x=前向，y=左向，z=上向）
下的 3D 点集。

所得点云是 BEV 高度图的构建素材：每相机深度图独立反投影后，
由 ``bev_height_builder.py`` 合并并按格子分箱。

兼容性：Python 3.8 / numpy 1.19.5。
"""
from __future__ import annotations

import numpy as np

from .osz_config import MAX_METRIC_DEPTH_M, Z_MAX_M, Z_MIN_M


def depth_map_to_ego_points(
    depth_map: np.ndarray,
    K: np.ndarray,
    T_cam2ego: np.ndarray,
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
    max_depth: float = MAX_METRIC_DEPTH_M,
) -> np.ndarray:
    """将度量深度图反投影为 ego 系 3D 点。

    Parameters
    ----------
    depth_map : np.ndarray
        ``(H, W)`` float32 度量深度（米），即相机系的 z 值。
    K : np.ndarray
        ``(3, 3)`` 相机内参矩阵。
    T_cam2ego : np.ndarray
        ``(4, 4)`` 相机 -> ego 外参变换。
    z_min : float, optional
        剔除低于此高度的 ego 系点（地面过滤）。
    z_max : float, optional
        剔除高于此高度的 ego 系点（建筑/天空过滤）。
    max_depth : float, optional
        剔除超过此距离的点（米）。

    Returns
    -------
    np.ndarray
        ``(N, 3)`` float32 ego 系点（x=前向，y=左向，z=上向）。
    """
    H, W = depth_map.shape

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    u = u.ravel()
    v = v.ravel()
    d = depth_map.ravel()

    valid = (d > 0.0) & (d < max_depth)
    u = u[valid]
    v = v[valid]
    d = d[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    pts_cam_h = np.concatenate(
        [pts_cam, np.ones((len(pts_cam), 1), dtype=np.float32)], axis=1
    )
    pts_ego = (T_cam2ego @ pts_cam_h.T).T[:, :3]

    pts_ego = pts_ego[(pts_ego[:, 2] >= z_min) & (pts_ego[:, 2] <= z_max)]
    return pts_ego.astype(np.float32)
