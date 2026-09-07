"""带稀疏 LiDAR 尺度对齐的 MiDaS v2.1 Small 深度估计器。

模型定义与 checkpoint 均从本地路径加载，避免 ``transformers``
依赖，兼容服务器的 torch 1.9.1+cu111 / Python 3.8 环境。

Notes
-----
- MiDaS 输出逆序式相对深度（值越大越近）。``align_to_lidar``
  同时尝试线性与逆序两种模型，逐图自动选择更契合数据的族。
- 当拟合出的尺度退化（接近零）时，回退到稳健的中值比率估计，
  避免产生恒定深度的幻影墙。
- 模型无法加载时，``MockDepthEstimator`` 原样返回 LiDAR 补全
  深度，使管线其余部分仍可运行。

兼容性：Python 3.8 / numpy 1.19.5 / torch 1.9.1 / Pillow 10.x。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .osz_config import MIDAS_MODEL_PATH, MIDAS_MODEL_URL, MIDAS_REPO_PATH


class DepthEstimator:
    """轻量单目深度估计器。

    Parameters
    ----------
    model_path : str, optional
        本地 MiDaS v2.1 Small checkpoint 路径。
    repo_path : str, optional
        包含 ``hubconf.py`` 的本地 MiDaS 仓库路径。
    device : str, optional
        ``'cpu'`` 或 ``'cuda'``。``None`` 时自动检测。

    Examples
    --------
    >>> est = DepthEstimator()
    >>> depth_metric = est.infer(image, lidar_sparse_depth=sparse_depth)
    """

    def __init__(
        self,
        model_path: str = MIDAS_MODEL_PATH,
        repo_path: str = MIDAS_REPO_PATH,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.repo_path = repo_path
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self._model = None
        self._transform = None

    @staticmethod
    def _cuda_available() -> bool:
        """检查 PyTorch CUDA 是否可用。"""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def load(self):
        """从本地仓库与本地 checkpoint 加载 MiDaS Small（幂等）。"""
        if self._model is not None:
            return self._model
        import torch

        repo_dir = Path(self.repo_path)
        checkpoint = Path(self.model_path)
        if not (repo_dir / "hubconf.py").exists():
            raise FileNotFoundError(
                "MiDaS 仓库缺失：{}。请将 isl-org/MiDaS 克隆到该处。"
                .format(repo_dir))
        if not checkpoint.exists():
            raise FileNotFoundError(
                "MiDaS checkpoint 缺失：{}（手动下载地址：{}）"
                .format(checkpoint, MIDAS_MODEL_URL))

        repo_abs = str(repo_dir.resolve())
        if repo_abs not in sys.path:
            sys.path.insert(0, repo_abs)
        model = torch.hub.load(repo_abs, "MiDaS_small", pretrained=False,
                               source="local")
        state = torch.load(str(checkpoint), map_location="cpu")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            # 兼容 DataParallel / 前缀包装过的 state_dict
            stripped = {}
            for key, value in state.items():
                if key.startswith("module."):
                    key = key[7:]
                elif key.startswith("model."):
                    key = key[6:]
                stripped[key] = value
            model.load_state_dict(stripped, strict=True)
        model.eval().to(self.device)
        transforms = torch.hub.load(repo_abs, "transforms", source="local")
        self._model = model
        self._transform = transforms.small_transform
        print("[DepthEstimator] 已在 {} 上加载本地 MiDaS v2.1 Small"
              .format(self.device))
        return self._model

    def infer_relative(
        self,
        image: np.ndarray,
        target_size: Optional[tuple] = None,
    ) -> np.ndarray:
        """对单张 RGB 图像预测相对深度图。

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB 图像。
        target_size : tuple, optional
            输出尺寸 ``(H, W)``。``None`` 时保持模型原生尺寸。

        Returns
        -------
        np.ndarray
            (H, W) float32 逆序式相对深度（值越大越近）。
        """
        import torch
        import torch.nn.functional as F

        model = self.load()
        input_t = self._transform(image).to(self.device)
        with torch.no_grad():
            pred = model(input_t).unsqueeze(1)
        depth_rel = F.interpolate(
            pred, size=image.shape[:2], mode="bicubic",
            align_corners=False)[0, 0].cpu().numpy().astype(np.float32)

        if target_size is not None:
            from PIL import Image as PILImage
            depth_rel = np.array(
                PILImage.fromarray(depth_rel).resize(
                    (target_size[1], target_size[0]), PILImage.BILINEAR),
                dtype=np.float32,
            )

        return depth_rel

    @staticmethod
    def align_to_lidar(
        depth_rel: np.ndarray,
        lidar_sparse: np.ndarray,
        min_points: int = 20,
        max_metric_depth: float = 70.0,
    ) -> np.ndarray:
        """拟合尺度与偏移，把相对深度转换为度量深度。

        在 LiDAR 像素上用最小二乘求解 ``metric = scale * rel + shift``。
        针对输出逆序式相对深度的 MiDaS（值越大越近），同时尝试
        ``metric = scale / rel + shift``。

        稳健性处理：

        - 拟合尺度退化（``|scale|`` 低于阈值）时，回退到中值比率
          估计，使深度仍随预测变化。
        - 输出裁剪到 ``[0, max_metric_depth]``，避免不真实的远墙。

        Parameters
        ----------
        depth_rel : np.ndarray
            (H, W) 相对深度图。
        lidar_sparse : np.ndarray
            (H, W) 稀疏 LiDAR 深度（米），0 表示无效。
        min_points : int, optional
            最小二乘拟合所需的最少 LiDAR 点数，不足时回退到
            中值比率缩放。
        max_metric_depth : float, optional
            输出深度的最大允许值（米）。

        Returns
        -------
        np.ndarray
            (H, W) float32 度量深度图。
        """
        valid = lidar_sparse > 0
        n_valid = valid.sum()
        if n_valid == 0:
            raise ValueError("没有可用于对齐的 LiDAR 深度。")

        rel_vals = depth_rel[valid]
        lidar_vals = lidar_sparse[valid]

        # 判定模型非退化的最小绝对尺度。
        # 线性：米 / 相对深度单位。逆序：米 * 相对深度单位。
        MIN_SCALE = {
            'linear': 0.5,
            'inverse': 1.0,
        }

        def _median_linear_scale():
            """返回稳健的线性尺度与零偏移。"""
            ratios = lidar_vals / (rel_vals + 1e-6)
            return float(np.median(ratios)), 0.0

        def _median_inverse_scale():
            """返回稳健的逆序尺度与零偏移。"""
            inv_ratios = lidar_vals * (rel_vals + 1e-6)
            return float(np.median(inv_ratios)), 0.0

        if n_valid >= min_points:
            # 先尝试线性模型：metric = scale * rel + shift
            A = np.stack([rel_vals, np.ones_like(rel_vals)], axis=1)
            scale, shift = np.linalg.lstsq(A, lidar_vals, rcond=None)[0]
            mode = 'linear'

            # MiDaS 输出逆序式相对深度（值越大越近）。若尺度为负，
            # 切换到逆序模型：metric = scale / rel + shift
            if scale < 0:
                inv_rel_vals = 1.0 / (rel_vals + 1e-6)
                A_inv = np.stack([inv_rel_vals, np.ones_like(inv_rel_vals)], axis=1)
                scale, shift = np.linalg.lstsq(A_inv, lidar_vals, rcond=None)[0]
                mode = 'inverse'

            # 很大的正偏移相当于恒定深度的地板：即使远处像素也会
            # 得到 metric >= shift，使中距占据虚高。两种族都要拒绝。
            if shift > 10.0:
                if mode == 'linear':
                    scale, shift = _median_linear_scale()
                    mode = 'linear_median'
                else:
                    scale, shift = _median_inverse_scale()
                    mode = 'inverse_median'

            # 尺度退化：scale ~= 0 意味着度量深度基本是常数偏移，
            # 会产生幻影远墙。回退到同族的稳健中值比率模型。
            if abs(scale) < MIN_SCALE[mode.replace('_median', '')]:
                if mode.startswith('linear'):
                    scale, shift = _median_linear_scale()
                else:
                    scale, shift = _median_inverse_scale()
                mode = f'{mode.replace("_median", "")}_median'
        else:
            # 回退：仅尺度，假设 shift = 0。先线性后逆序。
            scale, shift = _median_linear_scale()
            mode = 'linear'
            if scale < 0:
                scale, shift = _median_inverse_scale()
                mode = 'inverse'
            if abs(scale) < MIN_SCALE[mode]:
                mode = f'{mode}_median'

        if mode.startswith('linear'):
            depth_metric = scale * depth_rel + shift
        else:
            depth_metric = scale / (depth_rel + 1e-6) + shift

        depth_metric = np.clip(depth_metric, 0.0, max_metric_depth)
        print(f"[align_to_lidar] n_valid={n_valid}, {mode}: "
              f"scale={scale:.3f}, shift={shift:.3f}")
        return depth_metric.astype(np.float32)

    def infer(
        self,
        image: np.ndarray,
        lidar_sparse_depth: Optional[np.ndarray] = None,
        target_size: Optional[tuple] = None,
    ) -> np.ndarray:
        """预测度量深度图。

        给定 ``lidar_sparse_depth`` 时将相对深度对齐到度量尺度；
        否则返回原始相对深度（不能用于 ego 反投影）。

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB 图像。
        lidar_sparse_depth : np.ndarray, optional
            (H, W) 用于对齐的稀疏 LiDAR 深度。
        target_size : tuple, optional
            可选的输出尺寸 ``(H, W)``。

        Returns
        -------
        np.ndarray
            (H, W) float32 深度图。
        """
        depth_rel = self.infer_relative(image, target_size=target_size)

        if lidar_sparse_depth is not None:
            if lidar_sparse_depth.shape != depth_rel.shape:
                raise ValueError(
                    f"lidar_sparse_depth 形状 {lidar_sparse_depth.shape} "
                    f"与深度形状 {depth_rel.shape} 不匹配"
                )
            return self.align_to_lidar(depth_rel, lidar_sparse_depth)

        return depth_rel


class MockDepthEstimator:
    """无网络/模型时的占位深度估计器。

    把提供的 LiDAR 补全深度图原样作为"预测"度量深度返回，使图像
    -> BEV 管线的其余部分可以在没有真实单目深度模型的情况下
    构建与测试。

    Examples
    --------
    >>> est = MockDepthEstimator()
    >>> depth_metric = est.infer(image, lidar_dense_depth=dense_depth)
    """

    def __init__(self):
        pass

    def infer(
        self,
        image: np.ndarray,
        lidar_dense_depth: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """返回度量深度图。

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) uint8 RGB 图像（仅用于形状校验）。
        lidar_dense_depth : np.ndarray, optional
            (H, W) float32 度量深度。``None`` 时返回常数深度占位。

        Returns
        -------
        np.ndarray
            (H, W) float32 度量深度图。
        """
        H, W = image.shape[:2]
        if lidar_dense_depth is not None:
            if lidar_dense_depth.shape[:2] != (H, W):
                raise ValueError(
                    f"lidar_dense_depth 形状 {lidar_dense_depth.shape} "
                    f"与图像形状 {(H, W)} 不匹配"
                )
            return lidar_dense_depth.astype(np.float32)

        # 最后手段：常数 20 m 占位，保证下游代码可运行。
        print("[MockDepthEstimator] 警告：无深度输入，返回常数 20m。")
        return np.full((H, W), 20.0, dtype=np.float32)
