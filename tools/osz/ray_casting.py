"""高度感知 360° 射线投射（NumPy 后端）。

现行 OSZ 计算管线（与 ``run_export.py`` 默认路径一致）：

1. 每相机深度图只推理一次，反投影到 ego 系并逐格取最大高度，
   得到 BEV 高度图（相机主、LiDAR 兜底融合，见
   ``bev_height_builder.py``）。
2. 从 ego 出发向 360° 全方向投射射线，按高度阈值产生两层阴影：
   - ``osz_ground``：任何非空格子都挡住射线（地面层完全被遮挡）。
   - ``osz_eye``：只有高度超过观察者眼高的格子挡住射线。
   - 差集 ``osz_ground & ~osz_eye`` 即半透明区：地面被遮挡但
     上方体积可见。

坐标约定
--------
- 轴 0 = ego-x（前向），轴 1 = ego-y（左向）。
- ``bev_height[i, j]`` 中 ``i`` 为 ego-x 索引，``j`` 为 ego-y 索引。

兼容性：Python 3.8 / numpy 1.19.5。
"""
from __future__ import annotations

import numpy as np

from .osz_config import (
    EGO_CLEARANCE_RADIUS_M,
    OBSERVER_HEIGHT_M,
    Z_MAX_M,
    Z_MIN_M,
)


def cast_osz_height_aware(
    bev_height: np.ndarray,
    grid,
    observer_height: float = OBSERVER_HEIGHT_M,
    substep: float = 0.25,
):
    """在 BEV 高度图上进行 ego 中心 360° 射线投射。

    产生两层阴影掩码：

    - ``osz_ground``：任何占据格子都挡住射线（二值行为）。
    - ``osz_eye``：仅高度超过 ``observer_height`` 的格子挡住射线。

    差集 ``osz_ground & ~osz_eye`` 为半透明区：地面被遮挡但上方
    体积可见。

    Parameters
    ----------
    bev_height : np.ndarray
        (nx, ny) float32 每格最大高度（0 = 空格）。
    grid : BEVGrid
        提供 ``nx``、``ny``、``bev_range``、``bev_res`` 的网格实例。
    observer_height : float, optional
        观察者眼高（米）。
    substep : float, optional
        射线步长（BEV 格子数）。

    Returns
    -------
    osz_ground : np.ndarray
        (nx, ny) bool 地面层完全被遮挡的格子。
    osz_eye : np.ndarray
        (nx, ny) bool 眼高处被遮挡的格子。
    """
    nx, ny = grid.nx, grid.ny
    x_min, x_max, y_min, y_max = grid.bev_range

    ego_xi = int(np.floor((0.0 - x_min) / grid.bev_res))
    ego_yi = int(np.floor((y_max - 0.0) / grid.bev_res))

    osz_ground = np.zeros((nx, ny), dtype=bool)
    osz_eye = np.zeros((nx, ny), dtype=bool)

    if not (0 <= ego_xi < nx and 0 <= ego_yi < ny):
        return osz_ground, osz_eye

    # 清空自车周围的小半径区域，防止自遮挡。
    bev_height = bev_height.copy()
    radius_m = EGO_CLEARANCE_RADIUS_M
    radius_cells = radius_m / grid.bev_res
    xg, yg = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    dist_cells = np.sqrt((xg - ego_xi) ** 2 + (yg - ego_yi) ** 2)
    bev_height[dist_cells < radius_cells] = 0.0

    max_range_cells = max(nx, ny)
    n_angles = int(2 * np.pi * max_range_cells / substep)
    n_angles = max(n_angles, 720)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    dx = np.cos(angles) * substep
    dy = np.sin(angles) * substep
    x = np.full(n_angles, float(ego_xi))
    y = np.full(n_angles, float(ego_yi))

    hit_ground = np.zeros(n_angles, dtype=bool)
    hit_eye = np.zeros(n_angles, dtype=bool)
    active = np.ones(n_angles, dtype=bool)
    max_steps = int(max_range_cells / substep)

    for _ in range(max_steps):
        x[active] += dx[active]
        y[active] += dy[active]
        xi = np.rint(x).astype(np.int32)
        yi = np.rint(y).astype(np.int32)

        in_b = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        active &= in_b
        if not active.any():
            break

        idx = np.where(active)[0]
        xi_a, yi_a = xi[idx], yi[idx]
        h = bev_height[xi_a, yi_a]

        prev_g = hit_ground[idx]
        osz_ground[xi_a[prev_g], yi_a[prev_g]] = True
        hit_ground[idx] |= (h > 0.05)

        prev_e = hit_eye[idx]
        osz_eye[xi_a[prev_e], yi_a[prev_e]] = True
        hit_eye[idx] |= (h > observer_height)

    return osz_ground, osz_eye


def compute_osz_height_aware_from_cameras(
    cameras: dict,
    grid,
    estimator=None,
    observer_height: float = OBSERVER_HEIGHT_M,
    depth_key: str = "depth_map",
    use_uncertainty: bool = False,
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
):
    """由相机深度图端到端计算高度感知 OSZ（含 LiDAR 兜底）。

    Pipeline
    --------
    1. 构建融合 ``bev_height_max``（相机主，LiDAR 兜底）。
    2. 高度感知射线投射 -> ``osz_ground``、``osz_eye``。

    Parameters
    ----------
    cameras : dict
        ``{cam_name: {depth_map, K, T_cam2ego, image, ...}}``。
    grid : BEVGrid
        定义 BEV 网格的实例。
    estimator : object, optional
        可选深度估计器；提供时从各相机图像预测深度，否则使用
        ``depth_key``。
    observer_height : float, optional
        眼高（米）。
    depth_key : str, optional
        ``estimator`` 为 ``None`` 时使用的深度键。
    use_uncertainty : bool, optional
        ``True`` 时相机与 LiDAR 采用逆不确定性加权融合，否则硬兜底。
    z_min : float, optional
        地面过滤高度（米）。
    z_max : float, optional
        最大障碍物高度（米），超过该高度的点被丢弃。

    Returns
    -------
    bev_height : np.ndarray
        (nx, ny) float32 每格最大高度。
    osz_ground : np.ndarray
        (nx, ny) bool 地面层阴影。
    osz_eye : np.ndarray
        (nx, ny) bool 眼高阴影。
    """
    from .bev_height_builder import (
        build_bev_height_fused,
        build_bev_height_fused_uncertainty,
    )

    if use_uncertainty:
        bev_height = build_bev_height_fused_uncertainty(
            cameras, grid, estimator=estimator, depth_key=depth_key,
            z_min=z_min, z_max=z_max,
        )
    else:
        bev_height = build_bev_height_fused(
            cameras, grid, estimator=estimator, depth_key=depth_key,
            z_min=z_min, z_max=z_max,
        )

    bev_height = np.clip(bev_height, 0.0, z_max)
    osz_ground, osz_eye = cast_osz_height_aware(bev_height, grid, observer_height)
    return bev_height, osz_ground, osz_eye
