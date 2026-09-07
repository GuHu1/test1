"""逐帧 OSZ 编排：深度 -> ego 反投影 -> BEV 高度 -> 射线投射。

本模块是原 OSZMiner ``modules`` 管线在导出脚本中的逐帧编排层
（原 ``run_export.py`` 的 ``_prepare_working_depth`` 与
``_compute_osz``，外加盲区年龄递推与可行驶掩码求交）。几何实现
见 ``image_to_ego`` / ``bev_height_builder`` / ``ray_casting``；
本模块只做 NumPy 后端的调用编排。

兼容性：Python 3.8 / numpy 1.19.5。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .depth_estimator import MockDepthEstimator
from .osz_config import OBSERVER_HEIGHT_M, Z_MAX_M, Z_MIN_M
from .ray_casting import compute_osz_height_aware_from_cameras


def prepare_working_depth(cameras: dict, estimator) -> dict:
    """预测 OSZ 计算使用的逐相机度量深度。

    每相机只运行一次估计器，结果存入相机字典浅拷贝的
    ``depth_used`` 键。这样 OSZ 计算（以 ``estimator=None,
    depth_key='depth_used'`` 调用）与逐相机可视化消费的是完全相同
    的深度图。

    Parameters
    ----------
    cameras : dict
        loader 产出的逐相机数据字典。
    estimator : object 或 None
        深度估计器；``None`` 时直接使用相机既有的 ``depth_map``。

    Returns
    -------
    dict
        填好 ``depth_used`` 的 ``cameras`` 副本。
    """
    working = {name: dict(cam) for name, cam in cameras.items()}
    if estimator is None:
        for cam in working.values():
            cam['depth_used'] = cam.get('depth_map')
        return working

    use_mock = isinstance(estimator, MockDepthEstimator)
    for cam in working.values():
        if use_mock:
            depth = estimator.infer(
                cam['image'], lidar_dense_depth=cam.get('depth_map'))
        else:
            depth = estimator.infer(
                cam['image'],
                lidar_sparse_depth=cam.get('depth_map_sparse'),
                target_size=cam['image'].shape[:2])
        cam['depth_used'] = depth
    return working


def compute_osz(
    working_cameras: dict,
    grid,
    observer_height: float = OBSERVER_HEIGHT_M,
    use_uncertainty: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 NumPy 后端上计算 ``(bev_height, osz_ground, osz_eye)``。

    消费 :func:`prepare_working_depth` 填好的 ``depth_used`` 深度，
    依次执行 ego 反投影 -> 相机主 / LiDAR 兜底 BEV 高度融合 ->
    高度感知 360° 射线投射。

    Parameters
    ----------
    working_cameras : dict
        已填好 ``depth_used`` 的相机字典。
    grid : BEVGrid
        定义 BEV 网格的实例。
    observer_height : float, optional
        观察者眼高（米）。
    use_uncertainty : bool, optional
        ``True`` 时相机与 LiDAR 采用逆不确定性加权融合，否则硬兜底。

    Returns
    -------
    bev_height : np.ndarray
        (nx, ny) float32 每格最大高度。
    osz_ground : np.ndarray
        (nx, ny) bool 地面层阴影。
    osz_eye : np.ndarray
        (nx, ny) bool 眼高阴影。
    """
    return compute_osz_height_aware_from_cameras(
        working_cameras, grid, estimator=None, depth_key='depth_used',
        observer_height=observer_height, use_uncertainty=use_uncertainty,
        z_min=Z_MIN_M, z_max=Z_MAX_M)


def update_occlusion_age(
    age: np.ndarray,
    osz_eye: np.ndarray,
    dt: float,
) -> np.ndarray:
    """盲区年龄递推：眼高盲区内累加 ``dt``，可见即归零。

    Parameters
    ----------
    age : np.ndarray
        (nx, ny) float32 上一帧的盲区年龄（秒）。
    osz_eye : np.ndarray
        (nx, ny) bool 当前帧眼高盲区。
    dt : float
        递推步长（秒）。

    Returns
    -------
    np.ndarray
        (nx, ny) float32 更新后的盲区年龄。
    """
    return np.where(osz_eye, age + dt, 0.0).astype(np.float32)


def apply_drivable_mask(
    osz_ground: np.ndarray,
    osz_eye: np.ndarray,
    semi: np.ndarray,
    age: np.ndarray,
    drivable: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将 OSZ 各通道与可行驶区域掩码求交。

    Parameters
    ----------
    osz_ground, osz_eye, semi : np.ndarray
        (nx, ny) bool 各 OSZ 通道。
    age : np.ndarray
        (nx, ny) float32 盲区年龄。
    drivable : np.ndarray
        (nx, ny) bool 可行驶区域掩码。

    Returns
    -------
    tuple
        求交后的 ``(osz_ground, osz_eye, semi, age)``。
    """
    osz_ground = osz_ground & drivable
    osz_eye = osz_eye & drivable
    semi = semi & drivable
    age = age * drivable.astype(np.float32)
    return osz_ground, osz_eye, semi, age
