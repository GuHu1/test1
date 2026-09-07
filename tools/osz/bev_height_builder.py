"""由 ego 系 3D 点构建高度感知 BEV 网格。

本模块把逐相机与 LiDAR 的深度测量聚合为鸟瞰图（BEV）高度图。
支持相机主、LiDAR 兜底的硬融合，以及可选的不确定性加权融合。

兼容性：Python 3.8 / numpy 1.19.5。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .osz_config import MAX_METRIC_DEPTH_M, Z_MAX_M, Z_MIN_M


def build_bev_height_max(
    pts_ego: np.ndarray,
    grid,
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
) -> np.ndarray:
    """由 ego 系点计算每格最大高度。

    对每个 BEV 格子存落入该点的最大 z 坐标（高度）。空格子为 0。

    Parameters
    ----------
    pts_ego : np.ndarray
        (N, 3) float32 ego 系点，已完成地面过滤。
    grid : BEVGrid
        提供 ``bev_range``、``nx``、``ny``、``bev_res`` 的网格实例。
    z_min : float, optional
        考虑的最小高度（额外的安全过滤）。
    z_max : float, optional
        考虑的最大高度（建筑/天空过滤）。

    Returns
    -------
    np.ndarray
        (nx, ny) float32 每格最大高度。0 表示空格。
    """
    x_min, x_max, y_min, y_max = grid.bev_range
    nx, ny = grid.nx, grid.ny
    res = grid.bev_res

    xi = np.floor((pts_ego[:, 0] - x_min) / res).astype(np.int32)
    yi = np.floor((y_max - pts_ego[:, 1]) / res).astype(np.int32)
    zi = pts_ego[:, 2]

    valid = ((xi >= 0) & (xi < nx) &
             (yi >= 0) & (yi < ny) &
             (zi >= z_min) & (zi <= z_max))
    xi, yi, zi = xi[valid], yi[valid], zi[valid]

    bev_height = np.zeros((nx, ny), dtype=np.float32)
    np.maximum.at(bev_height, (xi, yi), zi)
    return bev_height


def _prepare_camera_depths(
    cameras: dict,
    estimator=None,
    depth_key: str = "depth_map",
) -> dict:
    """由估计器或既有键构建相机主深度图。

    Parameters
    ----------
    cameras : dict
        相机名 -> 相机数据字典。
    estimator : object, optional
        从相机图像预测深度的估计器。``None`` 时直接使用既有的
        ``depth_key`` 深度图。
    depth_key : str, optional
        ``estimator`` 为 ``None`` 时从 ``cameras`` 读取的深度键。

    Returns
    -------
    dict
        ``{cam_name: {'depth_map': (H, W), 'K': (3, 3), 'T_cam2ego': (4, 4)}}``。
    """
    if estimator is not None:
        from .depth_estimator import MockDepthEstimator
        use_mock = isinstance(estimator, MockDepthEstimator)
        cam_depths = {}
        for cam_name, cam_data in cameras.items():
            if use_mock:
                pred = estimator.infer(
                    cam_data['image'],
                    lidar_dense_depth=cam_data.get('depth_map'),
                )
            else:
                pred = estimator.infer(
                    cam_data['image'],
                    lidar_sparse_depth=cam_data.get('depth_map_sparse'),
                )
            cam_depths[cam_name] = {
                'depth_map': pred,
                'K': cam_data['K'],
                'T_cam2ego': cam_data['T_cam2ego'],
            }
    else:
        cam_depths = {
            n: {'depth_map': cameras[n][depth_key],
                'K': cameras[n]['K'],
                'T_cam2ego': cameras[n]['T_cam2ego']}
            for n in cameras
        }
    return cam_depths


def _prepare_lidar_fallback_depths(cameras: dict) -> dict:
    """构建 LiDAR 兜底深度图，稀疏优先于补全。"""
    return {
        cam_name: {
            'depth_map': cam_data.get('depth_map_sparse', cam_data.get('depth_map')),
            'K': cam_data['K'],
            'T_cam2ego': cam_data['T_cam2ego'],
        }
        for cam_name, cam_data in cameras.items()
    }


def build_bev_height_fused(
    cameras: dict,
    grid,
    estimator=None,
    depth_key: str = "depth_map",
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
) -> np.ndarray:
    """相机主 + LiDAR 兜底融合的 BEV 高度图。

    Strategy
    --------
    - 提供学习型深度估计器时，逐相机从图像预测深度（相机主模式）。
    - 否则使用既有相机 ``depth_map``（通常为 LiDAR 补全结果）。
    - 由 ``depth_map_sparse``/``depth_map`` 构建 LiDAR-only 高度图
      作为兜底。
    - 每格优先使用相机预测；相机为空处用 LiDAR 测量填充。

    Parameters
    ----------
    cameras : dict
        ``{cam_name: {depth_map, depth_map_sparse, K, T_cam2ego, image}}``。
    grid : BEVGrid
        定义 BEV 网格的实例。
    estimator : object, optional
        相机主预测的可选深度估计器。
    depth_key : str, optional
        ``estimator`` 为 ``None`` 时使用的相机深度键。
    z_min : float, optional
        地面过滤高度（米）。
    z_max : float, optional
        最大障碍物高度（米），超过该高度的点被丢弃。

    Returns
    -------
    np.ndarray
        (nx, ny) float32 融合后的每格最大高度。
    """
    # 相机主高度图
    cam_depths = _prepare_camera_depths(cameras, estimator=estimator, depth_key=depth_key)
    bev_height_cam = build_bev_height_from_cameras(
        cam_depths, grid, depth_key='depth_map', z_min=z_min, z_max=z_max
    )

    # LiDAR 兜底高度图：用原始稀疏 LiDAR 投影，以便在学习/补全的
    # 相机深度为空处仍有填充。
    lidar_depths = _prepare_lidar_fallback_depths(cameras)
    bev_height_lidar = build_bev_height_from_cameras(
        lidar_depths, grid, depth_key='depth_map', z_min=z_min, z_max=z_max
    )

    # 兜底：相机有值处用相机，相机为空处用 LiDAR
    bev_height_fused = bev_height_cam.copy()
    lidar_fallback = (bev_height_fused <= 0.05) & (bev_height_lidar > 0.05)
    bev_height_fused[lidar_fallback] = bev_height_lidar[lidar_fallback]

    n_fallback = int(lidar_fallback.sum())
    n_total = bev_height_fused.size
    print(
        f"[bev_height_fused] camera cells={(bev_height_cam > 0.05).sum()}, "
        f"lidar cells={(bev_height_lidar > 0.05).sum()}, "
        f"fallback cells={n_fallback} ({n_fallback / max(n_total, 1) * 100:.2f}%)"
    )

    return bev_height_fused


def build_bev_height_from_cameras(
    cameras: dict,
    grid,
    depth_key: str = "depth_map",
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
) -> np.ndarray:
    """反投影多相机深度图构建 BEV 高度图。

    Parameters
    ----------
    cameras : dict
        ``{cam_name: {'depth_map': (H, W), 'K': (3, 3), 'T_cam2ego': (4, 4)}}``。
    grid : BEVGrid
        定义 BEV 网格的实例。
    depth_key : str, optional
        使用的深度键。``'depth_map'`` 为补全/预测深度，
        ``'depth_map_sparse'`` 为原始稀疏 LiDAR 深度。
    z_min : float, optional
        地面过滤高度（米）。
    z_max : float, optional
        最大障碍物高度（米），超过该高度的点被丢弃。

    Returns
    -------
    np.ndarray
        (nx, ny) float32 每格最大高度。
    """
    from .image_to_ego import depth_map_to_ego_points

    bev_height = np.zeros((grid.nx, grid.ny), dtype=np.float32)

    for cam_name, cam_data in cameras.items():
        depth_map = cam_data[depth_key]
        K = cam_data['K']
        T_cam2ego = cam_data['T_cam2ego']

        pts_ego = depth_map_to_ego_points(depth_map, K, T_cam2ego, z_min=z_min, z_max=z_max)
        if len(pts_ego) == 0:
            continue

        cam_height = build_bev_height_max(pts_ego, grid, z_min=z_min, z_max=z_max)
        np.maximum(bev_height, cam_height, out=bev_height)

    return bev_height


def build_bev_height_and_uncertainty_from_cameras(
    cameras: dict,
    grid,
    depth_key: str = "depth_map",
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
    uncertainty_mode: str = "depth",
    max_depth: float = MAX_METRIC_DEPTH_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建逐格高度、平均不确定性与点密度图。

    Parameters
    ----------
    cameras : dict
        ``{cam_name: {'depth_map': (H, W), 'K': (3, 3), 'T_cam2ego': (4, 4)}}``。
    grid : BEVGrid
        定义 BEV 网格的实例。
    depth_key : str, optional
        使用的深度键。
    z_min : float, optional
        地面过滤高度（米）。
    z_max : float, optional
        最大障碍物高度（米），超过该高度的点被丢弃。
    uncertainty_mode : {'depth', 'constant'}, optional
        ``'depth'`` 使不确定性与 ego xy 距离成正比（越远越嘈杂）。
        ``'constant'`` 给每点相同的不确定性（用于仅按 LiDAR 密度
        加权）。
    max_depth : float, optional
        ``uncertainty_mode='depth'`` 时的距离归一化因子。

    Returns
    -------
    bev_height : np.ndarray
        (nx, ny) float32 每格最大高度。
    bev_uncertainty : np.ndarray
        (nx, ny) float32 平均点不确定性（空格为 ``inf``）。
    bev_density : np.ndarray
        (nx, ny) float32 每格点数。
    """
    from .image_to_ego import depth_map_to_ego_points

    nx, ny = grid.nx, grid.ny
    bev_height = np.zeros((nx, ny), dtype=np.float32)
    bev_unc_sum = np.zeros((nx, ny), dtype=np.float32)
    bev_density = np.zeros((nx, ny), dtype=np.float32)

    x_min, x_max, y_min, y_max = grid.bev_range

    for cam_name, cam_data in cameras.items():
        depth_map = cam_data[depth_key]
        K = cam_data['K']
        T_cam2ego = cam_data['T_cam2ego']

        pts_ego = depth_map_to_ego_points(
            depth_map, K, T_cam2ego, z_min=z_min, z_max=z_max, max_depth=max_depth
        )
        if len(pts_ego) == 0:
            continue

        if uncertainty_mode == "depth":
            dist = np.linalg.norm(pts_ego[:, :2], axis=1)
            point_unc = dist / max_depth
        else:
            point_unc = np.ones(len(pts_ego), dtype=np.float32)

        xi = np.floor((pts_ego[:, 0] - x_min) / grid.bev_res).astype(np.int32)
        yi = np.floor((y_max - pts_ego[:, 1]) / grid.bev_res).astype(np.int32)
        zi = pts_ego[:, 2]

        valid = ((xi >= 0) & (xi < nx) &
                 (yi >= 0) & (yi < ny) &
                 (zi >= z_min) & (zi <= z_max))
        xi, yi, zi, point_unc = xi[valid], yi[valid], zi[valid], point_unc[valid]

        np.maximum.at(bev_height, (xi, yi), zi)
        np.add.at(bev_unc_sum, (xi, yi), point_unc)
        np.add.at(bev_density, (xi, yi), 1.0)

    bev_uncertainty = np.full((nx, ny), np.inf, dtype=np.float32)
    valid_cells = bev_density > 0
    bev_uncertainty[valid_cells] = bev_unc_sum[valid_cells] / bev_density[valid_cells]

    return bev_height, bev_uncertainty, bev_density


def build_bev_height_fused_uncertainty(
    cameras: dict,
    grid,
    estimator=None,
    depth_key: str = "depth_map",
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
) -> np.ndarray:
    """``bev_height_max`` 的不确定性感知相机-LiDAR 融合。

    Strategy
    --------
    - 相机深度预测主高度，其不确定性随距离增大。
    - LiDAR 稀疏深度提供兜底与补充高度，其不确定性与局部点密度
      成反比。
    - 双方均有测量处按逆不确定性加权融合。
    - 相机测量为空处回退到 LiDAR（与硬兜底一致）。

    Parameters
    ----------
    cameras : dict
        ``{cam_name: {depth_map, depth_map_sparse, K, T_cam2ego, image}}``。
    grid : BEVGrid
        定义 BEV 网格的实例。
    estimator : object, optional
        相机主预测的可选深度估计器。
    depth_key : str, optional
        ``estimator`` 为 ``None`` 时使用的相机深度键。
    z_min : float, optional
        地面过滤高度（米）。
    z_max : float, optional
        最大障碍物高度（米），超过该高度的点被丢弃。

    Returns
    -------
    np.ndarray
        (nx, ny) float32 融合后的每格高度。
    """
    # 相机主深度图
    cam_depths = _prepare_camera_depths(cameras, estimator=estimator, depth_key=depth_key)
    h_cam, unc_cam, density_cam = build_bev_height_and_uncertainty_from_cameras(
        cam_depths, grid, depth_key='depth_map', z_min=z_min, z_max=z_max,
        uncertainty_mode='depth'
    )

    # LiDAR 稀疏兜底
    lidar_depths = _prepare_lidar_fallback_depths(cameras)
    h_lidar, _, density_lidar = build_bev_height_and_uncertainty_from_cameras(
        lidar_depths, grid, depth_key='depth_map', z_min=z_min, z_max=z_max,
        uncertainty_mode='constant'
    )

    # LiDAR 不确定性：局部密度越高越可信。
    unc_lidar = 1.0 / (density_lidar + 1.0)

    eps = 1e-6
    w_cam = 1.0 / (unc_cam + eps)
    w_lidar = 1.0 / (unc_lidar + eps)

    cam_valid = h_cam > 0.05
    lidar_valid = h_lidar > 0.05
    both_valid = cam_valid & lidar_valid

    bev_height_fused = np.zeros_like(h_cam)

    # 仅相机
    bev_height_fused[cam_valid & ~lidar_valid] = h_cam[cam_valid & ~lidar_valid]

    # 仅 LiDAR（硬兜底）
    bev_height_fused[~cam_valid & lidar_valid] = h_lidar[~cam_valid & lidar_valid]

    # 双方都有：逆不确定性加权融合
    denom = w_cam[both_valid] + w_lidar[both_valid]
    bev_height_fused[both_valid] = (
        w_cam[both_valid] * h_cam[both_valid] +
        w_lidar[both_valid] * h_lidar[both_valid]
    ) / denom

    n_fallback = int((~cam_valid & lidar_valid).sum())
    n_weighted = int(both_valid.sum())
    n_total = bev_height_fused.size
    print(
        f"[bev_height_fused_uncertainty] camera cells={cam_valid.sum()}, "
        f"lidar cells={lidar_valid.sum()}, both={n_weighted}, "
        f"fallback cells={n_fallback} ({n_fallback / max(n_total, 1) * 100:.2f}%)"
    )

    return bev_height_fused
