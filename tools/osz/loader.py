"""nuScenes 样本加载器：产出 OSZ 管线的逐相机输入帧。

``NuScenesOSZLoader`` 遍历 nuScenes 样本，对每个样本聚合 LiDAR
（``lidar.aggregate_lidar_sweeps``）、投影到各相机并补全深度
（``lidar.densify_depth_map``），附带 RGB 图像、内参与外参。
nuScenes devkit 或数据根目录不可用时改产出合成 mock 帧，也可通过
``force_mock=True`` 显式启用。

兼容性：Python 3.8 / numpy 1.19.5 / nuscenes-devkit 1.1.9 /
matplotlib 3.5.3 / Pillow 10.x。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from matplotlib import cm

from .osz_config import MAX_METRIC_DEPTH_M, NUSCENES_CAMERAS
from .lidar import (
    NUSCENES_AVAILABLE,
    _get_transform,
    aggregate_lidar_sweeps,
    densify_depth_map,
    project_lidar_to_camera,
)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _get_intrinsic(nusc, cam_token: str) -> np.ndarray:
    """返回相机内参矩阵（3x3）。"""
    sample_data = nusc.get('sample_data', cam_token)
    cs = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    K = np.array(cs['camera_intrinsic'], dtype=np.float32)
    return K


class NuScenesOSZLoader:
    """遍历 nuScenes 样本并产出 OSZ 输入。

    对每个样本，loader 产出一个包含样本 token 与逐相机数据的字典：
    稠密深度图、稀疏深度图、RGB 图像、内参与相机->ego 变换。

    当 nuScenes devkit 或数据根目录不可用时，loader 改为产出合成
    mock 帧；也可通过 ``force_mock=True`` 显式启用。

    Parameters
    ----------
    dataroot : str, optional
        nuScenes 数据集根目录。默认 ``'/data/sets/nuscenes'``。
    version : str, optional
        nuScenes 数据划分版本。默认 ``'v1.0-mini'``。
    cameras : List[str] 或 None, optional
        要加载的相机名。默认 ``NUSCENES_CAMERAS``。
    max_samples : int 或 None, optional
        迭代的最大样本数。``None`` 迭代全量（mock 模式下为 3 帧）。
    img_h : int, optional
        输出图像高。默认 900。
    img_w : int, optional
        输出图像宽。默认 1600。
    n_sweeps : int, optional
        投影前聚合的历史 LiDAR 扫描数。默认 0。
    force_mock : bool, optional
        强制使用合成 mock 数据（本地冒烟测试用）。默认 False。

    Attributes
    ----------
    use_mock : bool
        当前是否运行在 mock 模式。

    Yields
    ------
    dict
        帧字典，键为：

        - ``'sample_token'`` (str)：nuScenes 样本 token。
        - ``'cameras'`` (dict)：相机名 -> 包含 ``'depth_map'``、
          ``'depth_map_sparse'``、``'image'``、``'K'``、
          ``'T_cam2ego'``、``'img_h'``、``'img_w'`` 的字典。

    Examples
    --------
    >>> loader = NuScenesOSZLoader(dataroot='/data/nuscenes',
    ...                            version='v1.0-mini')
    >>> for frame in loader:
    ...     print(frame['sample_token'])
    """

    def __init__(
        self,
        dataroot: str = '/data/sets/nuscenes',
        version: str = 'v1.0-mini',
        cameras: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        img_h: int = 900,
        img_w: int = 1600,
        n_sweeps: int = 0,
        force_mock: bool = False,
    ):
        """初始化 loader 并选择真实或 mock 数据源。"""
        self.cameras = cameras or NUSCENES_CAMERAS
        self.max_samples = max_samples
        self.img_h = img_h
        self.img_w = img_w
        self.n_sweeps = n_sweeps

        if force_mock:
            self.use_mock = True
            self.n_mock = max_samples or 3
        elif NUSCENES_AVAILABLE and Path(dataroot).exists():
            # 延迟导入：mock 模式下不需要 devkit，导入告警也只由
            # lidar 模块打印一次。
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
            self.samples = self.nusc.sample
            if max_samples:
                self.samples = self.samples[:max_samples]
            self.use_mock = False
        else:
            print(f"[INFO] 未在 {dataroot} 找到 nuScenes 数据，改用合成 mock。")
            self.use_mock = True
            self.n_mock = max_samples or 3

    def __len__(self):
        """返回帧数（真实样本数或 mock 帧数）。"""
        return self.n_mock if self.use_mock else len(self.samples)

    def __iter__(self):
        """从真实数据集或 mock 生成器产出帧。"""
        if self.use_mock:
            yield from self._mock_iter()
        else:
            yield from self._nuscenes_iter()

    def build_frame_for_token(self, sample_token: str) -> dict:
        """按 token 构建单帧字典（无需迭代）。

        这是 ``run_export.py`` 指定 ``--sample_token`` 时使用的按
        token 入口：直接查表取样本记录，无论初始化时是否填充过
        ``self.samples`` 都可用。

        Parameters
        ----------
        sample_token : str
            nuScenes 样本 token。

        Returns
        -------
        dict
            含 ``'sample_token'`` 与 ``'cameras'`` 键的帧字典。
            相机字典内容见类 docstring。
        """
        sample = self.nusc.get('sample', sample_token)
        frame = {'sample_token': sample['token'], 'cameras': {}}

        # 读取 LiDAR（ego 系）。
        # 投影到相机前先聚合当前 + 历史扫描以获得更密的点覆盖，
        # 改善遮挡/远处的深度补全质量与兜底可靠性。
        pts_ego = aggregate_lidar_sweeps(
            self.nusc, sample_token, n_sweeps=self.n_sweeps
        )

        # 逐相机投影
        for cam_name in self.cameras:
            if cam_name not in sample['data']:
                continue
            cam_token = sample['data'][cam_name]
            cam_sd = self.nusc.get('sample_data', cam_token)
            T_cam2ego = _get_transform(self.nusc, cam_sd)
            K = _get_intrinsic(self.nusc, cam_token)

            depth_sparse = project_lidar_to_camera(
                pts_ego, K, T_cam2ego,
                self.img_h, self.img_w,
            )
            depth_dense = densify_depth_map(depth_sparse)
            # 与模型预测使用相同的最大深度截断，下游模块不会看到
            # 100 m 的幻影墙。
            depth_dense = np.clip(depth_dense, 0.0, MAX_METRIC_DEPTH_M)

            # 读取相机图像
            image = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            if PIL_AVAILABLE:
                try:
                    img_path = Path(self.nusc.dataroot) / cam_sd['filename']
                    image = np.array(Image.open(img_path).convert('RGB'))
                except Exception:
                    pass

            frame['cameras'][cam_name] = {
                'depth_map': depth_dense,        # (H, W) 米，已补全
                'depth_map_sparse': depth_sparse,# (H, W) 原始稀疏
                'image':     image,              # (H, W, 3) RGB
                'K':         K,                  # (3, 3)
                'T_cam2ego': T_cam2ego,          # (4, 4)
                'img_h':     self.img_h,
                'img_w':     self.img_w,
            }

        return frame

    def _nuscenes_iter(self):
        """迭代已加载的 nuScenes 样本并产出帧。"""
        for sample in self.samples:
            yield self.build_frame_for_token(sample['token'])

    def _mock_iter(self):
        """nuScenes 数据不可用时产出合成 mock 帧。

        合成场景遵循 nuScenes ego 系约定（``x`` 前向、``y`` 左向、
        ``z`` 上向）与相机系约定（``z`` 光轴前向、``x`` 右向、
        ``y`` 下向）。场景包含一个位于自车前方约 12 m 的实体箱形
        遮挡物，以及其后方的背景物体。
        """
        rng = np.random.default_rng(42)

        K = np.array([
            [1266.4, 0,      816.0],
            [0,      1266.4, 491.5],
            [0,      0,      1.0 ],
        ], dtype=np.float32)

        # nuScenes 前向相机外参：
        # cam_z(前) -> ego_x(前)，cam_x(右) -> -ego_y，cam_y(下) -> -ego_z
        def make_cam2ego(yaw_deg: float, tx: float, ty: float, tz: float) -> np.ndarray:
            """按给定偏航角与平移返回相机 -> ego 变换。"""
            # 基础旋转：相机光轴 = ego 前向
            R_base = np.array([
                [ 0, 0, 1],   # cam_z -> ego_x
                [-1, 0, 0],   # cam_x -> -ego_y（相机右 = ego 右 = -ego 左）
                [ 0,-1, 0],   # cam_y -> -ego_z
            ], dtype=np.float32)
            # ego 系内的偏航旋转
            yaw = np.deg2rad(yaw_deg)
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0,            0,           1],
            ], dtype=np.float32)
            R = Rz @ R_base
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3,  3] = [tx, ty, tz]
            return T

        # 3 个前向相机，带偏航偏移
        cam_configs = [
            # (name,            yaw_deg, tx,  ty,  tz)
            ('CAM_FRONT',            0,  1.5,  0.0, 1.5),
            ('CAM_FRONT_LEFT',      55,  1.5,  0.5, 1.5),
            ('CAM_FRONT_RIGHT',    -55,  1.5, -0.5, 1.5),
        ]

        for i in range(self.n_mock):
            frame = {'sample_token': f'mock_{i:04d}', 'cameras': {}}

            ox = 12.0 + rng.uniform(-1.0, 1.0)   # 遮挡物 x（前向）
            oy = 1.5 + rng.uniform(-0.3, 0.3)   # 遮挡物 y（横向）

            for cam_name, yaw_deg, tx, ty, tz in cam_configs:
                if cam_name not in self.cameras:
                    continue
                T_cam2ego = make_cam2ego(yaw_deg, tx, ty, tz)

                # 遮挡物正面（稠密，朝向 ego）
                box_pts = []
                for dy in np.linspace(-1.5, 1.5, 50):
                    for dz in np.linspace(0.05, 1.75, 35):
                        box_pts.append([ox, oy + dy, dz])
                # 遮挡物侧壁
                for dx in np.linspace(0, 4.0, 25):
                    for dz in np.linspace(0.05, 1.75, 25):
                        box_pts.append([ox + dx, oy - 1.5, dz])
                        box_pts.append([ox + dx, oy + 1.5, dz])
                box_pts = np.array(box_pts, dtype=np.float32)

                # 地面
                xs_g = np.linspace(1, 50, 80)
                ys_g = np.linspace(-10, 10, 40)
                xx, yy = np.meshgrid(xs_g, ys_g)
                gnd = np.stack([xx.ravel(), yy.ravel(),
                                np.zeros(xx.size)], axis=1).astype(np.float32)

                # 遮挡物后方的背景物体（应落入阴影区）
                bg_pts = []
                for dx in np.linspace(0, 5, 20):
                    for dy in np.linspace(-1.2, 1.2, 20):
                        for dz in np.linspace(0.1, 1.6, 10):
                            bg_pts.append([ox + 5 + dx, oy + dy, dz])
                bg_pts = np.array(bg_pts, dtype=np.float32)

                pts_ego = np.concatenate([box_pts, gnd, bg_pts], axis=0)
                depth_sparse = project_lidar_to_camera(
                    pts_ego, K, T_cam2ego, self.img_h, self.img_w
                )
                depth_dense = densify_depth_map(depth_sparse)

                # 由稠密深度可视化生成合成参考图像
                depth_norm = depth_dense / (depth_dense.max() + 1e-6)
                image = (cm.viridis(depth_norm)[:, :, :3] * 255).astype(np.uint8)

                frame['cameras'][cam_name] = {
                    'depth_map': depth_dense,
                    'depth_map_sparse': depth_sparse,
                    'image':     image,
                    'K':         K,
                    'T_cam2ego': T_cam2ego,
                    'img_h':     self.img_h,
                    'img_w':     self.img_w,
                }

            yield frame
