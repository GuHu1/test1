"""OSZ 可视化：逐相机深度叠加图、BEV 六面板与 GT/解释图。

合并原 OSZMiner ``visualize`` 包（``camera_viz`` / ``osz_panels`` /
``gt_viz``）的全部落盘可视化：

- ``visualize_frame_cameras``：每帧六视角原图 ``{CAM}.jpg`` +
  对应 OSZ 深度叠加图 ``{CAM}_osz.png``（障碍高度带红色高亮）。
- ``save_osz_panels``：BEV 六面板汇总图（高度 / 地面盲区 / 眼高
  盲区 / 半透明区 / 叠加 / 统计文本）。
- ``save_gt_osz`` / ``save_osz_explained``：GT 框叠加与 OSZ 解释
  图（仅真实 nuScenes 数据）。

坐标约定
--------
- ego 系：``x`` 前向，``y`` 左向，``z`` 上向。
- BEV 数组 ``(nx, ny)``，``indexing='ij'``：``i`` = ego-x（轴 0），
  ``j`` = ego-y（轴 1）。
- ``imshow`` 使用 ``origin='lower'`` +
  ``extent=[y_max, y_min, x_min, x_max]`` 且不做转置。

兼容性：Python 3.8 / numpy 1.19.5 / matplotlib 3.5.3 /
nuscenes-devkit 1.1.9 / pyquaternion 0.9.9 / Pillow 10.x。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .osz_config import (
    BEV_RANGE_M,
    BEV_RESOLUTION_M,
    MAX_METRIC_DEPTH_M,
    Z_MAX_M,
    Z_MIN_M,
    bev_extent,
)
from .coords import (
    VEHICLE_CATEGORIES,
    bev_box_corners_ego,
    ego_pose_from_sample_data,
    get_map_name,
    get_vehicle_states_ego,
)


# ---------------------------------------------------------------------------
# 逐相机可视化：原始图像与对应的 OSZ 深度叠加图
# ---------------------------------------------------------------------------
def backproject_pixel_heights(
    depth_map: np.ndarray,
    K: np.ndarray,
    T_cam2ego: np.ndarray,
    max_depth: float = MAX_METRIC_DEPTH_M,
) -> np.ndarray:
    """返回度量深度图每个像素的 ego 系高度。

    与 :func:`image_to_ego.depth_map_to_ego_points` 的反投影数学
    一致，但保留逐像素 ego-z 而非过滤点，使结果可以直接叠加在
    图像上可视化。

    Parameters
    ----------
    depth_map : np.ndarray
        ``(H, W)`` float32 度量深度（米），0 = 无效。
    K : np.ndarray
        ``(3, 3)`` 相机内参矩阵。
    T_cam2ego : np.ndarray
        ``(4, 4)`` 相机 -> ego 外参变换。
    max_depth : float, optional
        大于等于该值的深度视为无效。

    Returns
    -------
    np.ndarray
        ``(H, W)`` float32 ego 系高度；深度无效处为 ``NaN``。
    """
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    d = depth_map.astype(np.float32)

    valid = (d > 0.0) & (d < max_depth)
    heights = np.full((H, W), np.nan, dtype=np.float32)
    if not valid.any():
        return heights

    uu, vv, dd = u[valid], v[valid], d[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (uu - cx) * dd / fx
    y_cam = (vv - cy) * dd / fy
    pts_cam = np.stack([x_cam, y_cam, dd], axis=1)
    pts_cam_h = np.concatenate(
        [pts_cam, np.ones((len(pts_cam), 1), dtype=np.float32)], axis=1
    )
    pts_ego = (T_cam2ego @ pts_cam_h.T).T[:, :3]
    heights[valid] = pts_ego[:, 2]
    return heights


def obstacle_height_mask(
    depth_map: np.ndarray,
    K: np.ndarray,
    T_cam2ego: np.ndarray,
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
    max_depth: float = MAX_METRIC_DEPTH_M,
) -> np.ndarray:
    """标记反投影 ego 高度落在障碍带内的像素。

    Parameters
    ----------
    depth_map : np.ndarray
        ``(H, W)`` 度量深度图（米）。
    K : np.ndarray
        ``(3, 3)`` 相机内参矩阵。
    T_cam2ego : np.ndarray
        ``(4, 4)`` 相机 -> ego 外参变换。
    z_min : float, optional
        障碍高度带的下界（米）。
    z_max : float, optional
        障碍高度带的上界（米）。
    max_depth : float, optional
        大于等于该值的深度视为无效。

    Returns
    -------
    np.ndarray
        ``(H, W)`` bool 障碍带像素掩码。
    """
    heights = backproject_pixel_heights(
        depth_map, K, T_cam2ego, max_depth=max_depth)
    return np.isfinite(heights) & (heights >= z_min) & (heights <= z_max)


def save_camera_image(image: np.ndarray, save_path: Union[str, Path]) -> None:
    """把原始相机图以 JPEG 写盘。

    Parameters
    ----------
    image : np.ndarray
        ``(H, W, 3)`` uint8 RGB 图像。
    save_path : Union[str, Path]
        目标 ``.jpg`` 路径。父目录自动创建。
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("保存相机图需要 PIL。")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        path, quality=90)


def plot_camera_osz(
    cam_name: str,
    image: np.ndarray,
    depth_map: np.ndarray,
    obstacle_mask: np.ndarray,
    save_path: Union[str, Path],
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
    max_depth: float = MAX_METRIC_DEPTH_M,
) -> None:
    """保存单相机 OSZ 图：原图与深度 + 障碍带并排。

    右侧面板显示 OSZ 管线实际消费的度量深度图（turbo 伪彩；无效
    像素为深灰），障碍高度带以半透明红色填充与红色轮廓高亮。

    Parameters
    ----------
    cam_name : str
        相机名，如 ``'CAM_FRONT'``。
    image : np.ndarray
        ``(H, W, 3)`` uint8 RGB 图像。
    depth_map : np.ndarray
        ``(H, W)`` 度量深度图（米）。
    obstacle_mask : np.ndarray
        来自 :func:`obstacle_height_mask` 的 ``(H, W)`` bool 掩码。
    save_path : Union[str, Path]
        目标 ``.png`` 路径。父目录自动创建。
    z_min : float, optional
        障碍高度带下界（用于标题）。
    z_max : float, optional
        障碍高度带上界（用于标题）。
    max_depth : float, optional
        选取伪彩上限时使用的深度饱和值。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax_raw = axes[0]
    ax_raw.imshow(image)
    ax_raw.set_title(f"{cam_name} | raw image", fontsize=11)
    ax_raw.axis("off")

    ax_depth = axes[1]
    valid = (depth_map > 0.0) & (depth_map < max_depth)
    depth_masked = np.ma.masked_where(~valid, depth_map)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#404040")
    if valid.any():
        vmax = float(np.percentile(depth_map[valid], 99))
        vmax = max(vmax, 5.0)
    else:
        vmax = max_depth
    im = ax_depth.imshow(depth_masked, cmap=cmap, vmin=0.0, vmax=vmax)
    ax_depth.imshow(
        obstacle_mask.astype(np.float32),
        cmap=ListedColormap([(0, 0, 0, 0.0), (1, 0, 0, 0.45)]),
        vmin=0, vmax=1,
    )
    if obstacle_mask.any():
        ax_depth.contour(
            obstacle_mask.astype(np.float32), levels=[0.5],
            colors="red", linewidths=0.6)
    ax_depth.set_title(
        f"{cam_name} | OSZ depth (m), red = obstacle band "
        f"[{z_min:.1f}, {z_max:.1f}] m",
        fontsize=11,
    )
    ax_depth.axis("off")
    fig.colorbar(im, ax=ax_depth, fraction=0.03, pad=0.02, label="depth (m)")

    fig.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def visualize_frame_cameras(
    frame: dict,
    depth_used: dict,
    out_dir: Union[str, Path],
    z_min: float = Z_MIN_M,
    z_max: float = Z_MAX_M,
) -> None:
    """保存一帧中每个相机的原图与 OSZ 叠加图。

    Parameters
    ----------
    frame : dict
        来自 :class:`loader.NuScenesOSZLoader` 的帧字典，含
        ``cameras``。
    depth_used : dict
        相机名 -> OSZ 计算实际使用的度量深度图。
    out_dir : Union[str, Path]
        接收 ``{CAM}.jpg`` 与 ``{CAM}_osz.png`` 的目录。
    z_min : float, optional
        障碍高度带下界。
    z_max : float, optional
        障碍高度带上界。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam_name, cam_data in frame["cameras"].items():
        depth = depth_used.get(cam_name, cam_data.get("depth_map"))
        if depth is None:
            continue
        save_camera_image(cam_data["image"], out_dir / f"{cam_name}.jpg")
        mask = obstacle_height_mask(
            depth, cam_data["K"], cam_data["T_cam2ego"],
            z_min=z_min, z_max=z_max)
        plot_camera_osz(
            cam_name, cam_data["image"], depth, mask,
            out_dir / f"{cam_name}_osz.png", z_min=z_min, z_max=z_max)


# ---------------------------------------------------------------------------
# BEV 六面板汇总图
# ---------------------------------------------------------------------------
def save_osz_panels(
    bev_height: np.ndarray,
    osz_ground: np.ndarray,
    osz_eye: np.ndarray,
    semi: np.ndarray,
    occlusion_age: np.ndarray,
    grid,
    observer_height: float,
    sample_token: str,
    save_path: Union[str, Path],
    drivable_mask: Optional[np.ndarray] = None,
) -> None:
    """渲染并保存高度感知 OSZ 的六面板汇总图。

    Parameters
    ----------
    bev_height : np.ndarray
        BEV 高度图。
    osz_ground : np.ndarray
        二值地面 OSZ 掩码。
    osz_eye : np.ndarray
        二值眼高 OSZ 掩码。
    semi : np.ndarray
        二值半透明区掩码（ground 减去 eye）。
    occlusion_age : np.ndarray
        逐格盲区年龄（秒）。
    grid : BEVGrid
        产出 OSZ 掩码的网格实例（用于 extent / 分辨率）。
    observer_height : float
        观察者眼高（米）。
    sample_token : str
        图标题中显示的样本 token。
    save_path : Union[str, Path]
        目标 ``.png`` 路径。父目录自动创建。
    drivable_mask : Optional[np.ndarray], optional
        可行驶区域掩码；未过滤时为 ``None``。
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    title_filter = " | drivable-filtered" if drivable_mask is not None else ""
    fig.suptitle(
        f"Height-Aware OSZ | token={sample_token} | "
        f"observer={observer_height}m{title_filter}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    extent, xlim, ylim = bev_extent(grid.bev_range)

    # 第 0 行
    ax_h = axes[0, 0]
    im_h = ax_h.imshow(
        bev_height, origin="lower", extent=extent,
        cmap="viridis", vmin=0, vmax=Z_MAX_M,
    )
    ax_h.set_xlim(*xlim)
    ax_h.set_ylim(*ylim)
    ax_h.set_title("BEV height max", fontsize=11)
    ax_h.set_xlabel("y (m)", fontsize=8)
    ax_h.set_ylabel("x (m)", fontsize=8)
    plt.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04, label="height (m)")

    ax_g = axes[0, 1]
    ax_g.imshow(
        bev_height, origin="lower", extent=extent,
        cmap="Greys", vmin=0, vmax=Z_MAX_M, alpha=0.3,
    )
    ax_g.imshow(
        osz_ground, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#d32f2f"]), vmin=0, vmax=1, alpha=0.85,
    )
    ax_g.set_xlim(*xlim)
    ax_g.set_ylim(*ylim)
    ax_g.set_title(f"OSZ ground (binary) | {osz_ground.sum()} cells", fontsize=11)
    ax_g.set_xlabel("y (m)", fontsize=8)
    ax_g.set_ylabel("x (m)", fontsize=8)

    ax_e = axes[0, 2]
    ax_e.imshow(
        bev_height, origin="lower", extent=extent,
        cmap="Greys", vmin=0, vmax=Z_MAX_M, alpha=0.3,
    )
    ax_e.imshow(
        osz_eye, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#7b1fa2"]), vmin=0, vmax=1, alpha=0.85,
    )
    ax_e.set_xlim(*xlim)
    ax_e.set_ylim(*ylim)
    ax_e.set_title(f"OSZ eye (h > {observer_height}m) | {osz_eye.sum()} cells", fontsize=11)
    ax_e.set_xlabel("y (m)", fontsize=8)
    ax_e.set_ylabel("x (m)", fontsize=8)

    # 第 1 行
    ax_s = axes[1, 0]
    ax_s.imshow(
        bev_height, origin="lower", extent=extent,
        cmap="Greys", vmin=0, vmax=Z_MAX_M, alpha=0.3,
    )
    ax_s.imshow(
        semi, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#ff9800"]), vmin=0, vmax=1, alpha=0.85,
    )
    ax_s.set_xlim(*xlim)
    ax_s.set_ylim(*ylim)
    ax_s.set_title(f"Semi-transparent zone | {semi.sum()} cells", fontsize=11)
    ax_s.set_xlabel("y (m)", fontsize=8)
    ax_s.set_ylabel("x (m)", fontsize=8)

    ax_c = axes[1, 1]
    overlay = np.zeros((*bev_height.shape, 3))
    h_norm = np.clip(bev_height / Z_MAX_M, 0.0, 1.0)
    overlay[:, :, 0] = h_norm
    overlay[:, :, 1] = h_norm
    overlay[:, :, 2] = h_norm
    ax_c.imshow(overlay, origin="lower", extent=extent)
    ax_c.imshow(
        osz_ground, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#d32f2f"]), vmin=0, vmax=1, alpha=0.5,
    )
    ax_c.imshow(
        semi, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#ff9800"]), vmin=0, vmax=1, alpha=0.6,
    )
    ax_c.imshow(
        osz_eye, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#7b1fa2"]), vmin=0, vmax=1, alpha=0.7,
    )
    ax_c.set_xlim(*xlim)
    ax_c.set_ylim(*ylim)
    ax_c.set_title("Combined: red=ground, orange=semi, purple=eye", fontsize=11)
    ax_c.set_xlabel("y (m)", fontsize=8)
    ax_c.set_ylabel("x (m)", fontsize=8)

    ax_stats = axes[1, 2]
    ax_stats.axis("off")
    total = bev_height.size
    age_max = float(occlusion_age.max()) if occlusion_age.size else 0.0
    age_mean = float(occlusion_age[osz_eye].mean()) if osz_eye.any() else 0.0
    stats_text = (
        f"BEV grid: {grid.nx} x {grid.ny} = {total} cells\n"
        f"BEV resolution: {grid.bev_res} m\n"
        f"Observer height: {observer_height} m\n\n"
        f"Occupied cells: {(bev_height > 0).sum()} "
        f"({(bev_height > 0).sum() / total * 100:.1f}%)\n"
        f"OSZ ground cells: {osz_ground.sum()} "
        f"({osz_ground.sum() / total * 100:.1f}%)\n"
        f"OSZ eye cells: {osz_eye.sum()} "
        f"({osz_eye.sum() / total * 100:.1f}%)\n"
        f"Semi-transparent cells: {semi.sum()} "
        f"({semi.sum() / total * 100:.1f}%)\n\n"
        f"Occlusion age: max={age_max:.1f} s\n"
        f"  mean over eye cells={age_mean:.1f} s\n\n"
        f"Eye reduction vs ground:\n"
        f"  {osz_eye.sum() / max(osz_ground.sum(), 1) * 100:.1f}% of ground shadow\n"
        f"  diff = {osz_ground.sum() - osz_eye.sum()} cells"
    )
    ax_stats.text(
        0.1, 0.5, stats_text, transform=ax_stats.transAxes,
        fontsize=11, verticalalignment="center", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# GT 框叠加与 OSZ 解释可视化（仅真实 nuScenes 数据）
# ---------------------------------------------------------------------------
#: BEV 叠加图调色板（RGB 浮点）。
_OSZ_PALETTE = {
    "road": (0.29, 0.29, 0.29),
    "grass": (0.30, 0.69, 0.31),
    "obstacle": (1.00, 0.60, 0.00),
    "osz": (0.00, 0.00, 0.00),
    "ego": (0.10, 0.46, 0.82),
    "lane": (1.00, 1.00, 1.00),
    "text": (0.13, 0.13, 0.13),
    "text_mid": (0.33, 0.33, 0.33),
}

_VEHICLE_CATS = VEHICLE_CATEGORIES


def _draw_ego(ax: plt.Axes, size: float = 2.0, color: str = "#1976d2") -> None:
    """把自车画成带前向箭头的小矩形。"""
    rect = plt.Rectangle(
        (-size / 2, -size), size, size * 2,
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.85,
    )
    ax.add_patch(rect)
    ax.annotate(
        "", xy=(0, size * 1.5), xytext=(0, size),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )


def plot_gt_osz(
    osz_pa: np.ndarray,
    bev_occ: np.ndarray,
    drivable_mask: Optional[np.ndarray],
    nusc,
    sample_token: str,
    bev_range: Tuple[float, float, float, float] = BEV_RANGE_M,
    bev_res: float = BEV_RESOLUTION_M,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """在单面板 BEV 图上绘制 GT 框与 PA 相关 OSZ。

    幻影候选判定：某 GT 车辆的足印格子完全不落在占据图上（遮挡
    物本身不应在自己的阴影里），且其中心格位于 OSZ 内。该类框以
    红色描边标出，是"被遮挡的真实车辆"的直接证据。

    Parameters
    ----------
    osz_pa : np.ndarray
        OSZ 掩码 ``(nx, ny)``；已按可行驶区域过滤时传过滤后的
        ground OSZ，未过滤时传原始 ground OSZ。
    bev_occ : np.ndarray
        BEV 占据 / 遮挡物掩码 ``(nx, ny)``。
    drivable_mask : np.ndarray 或 None
        可行驶区域掩码；``None`` 时背景退化为"占据之外皆道路"。
    nusc : NuScenes
        已初始化的 nuScenes 数据集实例（用于取 GT 框）。
    sample_token : str
        nuScenes 样本 token。
    bev_range : Tuple[float, float, float, float], optional
        BEV 范围 ``(x_min, x_max, y_min, y_max)``，单位米。
    bev_res : float, optional
        BEV 格子边长（米）。
    save_path : str 或 None, optional
        给定时保存 PNG 到该路径。

    Returns
    -------
    matplotlib.figure.Figure
        生成的图；调用方负责关闭（或改用 :func:`save_gt_osz`）。
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    x_min, x_max, y_min, y_max = bev_range
    extent, xlim, ylim = bev_extent(bev_range)
    nx, ny = osz_pa.shape

    # 背景层：道路 / 不可行驶地面 / 遮挡物 / OSZ 阴影
    overlay = np.zeros((nx, ny, 3), dtype=np.float32)
    if drivable_mask is not None and drivable_mask.any():
        non_drivable = ~(drivable_mask | bev_occ)
        overlay[non_drivable] = _OSZ_PALETTE["grass"]
        overlay[drivable_mask] = _OSZ_PALETTE["road"]
    else:
        overlay[:] = _OSZ_PALETTE["grass"]
        overlay[~bev_occ] = _OSZ_PALETTE["road"]

    overlay[bev_occ] = _OSZ_PALETTE["obstacle"]
    if drivable_mask is not None and drivable_mask.any():
        overlay[osz_pa & drivable_mask] = _OSZ_PALETTE["osz"]
    else:
        overlay[osz_pa] = _OSZ_PALETTE["osz"]

    ax.imshow(overlay, origin="lower", extent=extent)

    # GT 框：逐框统计足印格与占据重叠，判定 in_osz
    boxes = get_vehicle_states_ego(nusc, sample_token, bev_range)

    for box in boxes:
        w, l = box["width"], box["length"]
        cos_h, sin_h = np.cos(box["yaw"]), np.sin(box["yaw"])
        half_local = np.array(
            [[l / 2, w / 2], [l / 2, -w / 2],
             [-l / 2, -w / 2], [-l / 2, w / 2]],
            dtype=np.float32,
        )
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        corners = (R @ half_local.T).T + np.array(
            [box["cx"], box["cy"]], dtype=np.float32
        )

        i_lo = max(0, int(np.floor((corners[:, 0].min() - x_min) / bev_res)))
        i_hi = min(nx - 1, int(np.ceil((corners[:, 0].max() - x_min) / bev_res)))
        j_lo = max(0, int(np.floor((y_max - corners[:, 1].max()) / bev_res)))
        j_hi = min(ny - 1, int(np.ceil((y_max - corners[:, 1].min()) / bev_res)))

        Rt = R.T
        total = 0
        n_in_occ = 0
        for i in range(i_lo, i_hi + 1):
            x_c = x_min + (i + 0.5) * bev_res
            for j in range(j_lo, j_hi + 1):
                y_c = y_max - (j + 0.5) * bev_res
                dx = x_c - box["cx"]
                dy = y_c - box["cy"]
                lx = Rt[0, 0] * dx + Rt[0, 1] * dy
                ly = Rt[1, 0] * dx + Rt[1, 1] * dy
                if abs(lx) > l / 2 or abs(ly) > w / 2:
                    continue
                total += 1
                if bev_occ[i, j]:
                    n_in_occ += 1

        # 中心格索引裁剪到网格内，避免恰在范围边界的车辆越界。
        ci = int(np.rint((box["cx"] - x_min) / bev_res))
        cj = int(np.rint((y_max - box["cy"]) / bev_res))
        ci = min(max(ci, 0), nx - 1)
        cj = min(max(cj, 0), ny - 1)
        box["in_osz"] = (
            (n_in_occ == 0) and (total > 0) and bool(osz_pa[ci, cj])
        )
        box["occ_overlap"] = n_in_occ / max(total, 1)
        box["footprint_cells"] = total

    n_phantom = sum(
        1 for b in boxes if b["in_osz"] and b["category"] in _VEHICLE_CATS
    )

    for box in boxes:
        # 逐框重算航向三角（不复用上一循环残留值，箭头方向才正确）。
        cos_h, sin_h = np.cos(box["yaw"]), np.sin(box["yaw"])
        corners = bev_box_corners_ego(
            box["cx"], box["cy"], box["yaw"], box["width"], box["length"]
        )
        poly_ax_x = list(corners[:, 1]) + [corners[0, 1]]  # ego-y -> 横轴
        poly_ax_y = list(corners[:, 0]) + [corners[0, 0]]  # ego-x -> 纵轴

        cat = box["category"]
        if cat in _VEHICLE_CATS:
            if box["in_osz"]:
                base_color = "#ff0000"
                edge_color = "#ffffff"
                facecolor = "none"
                lw = 2.2
                alpha = 1.0
            else:
                base_color = "#ff9800"
                edge_color = "#e65100"
                facecolor = "#ff9800"
                lw = 1.8
                alpha = 0.45
        else:
            base_color = "#AAAAAA"
            edge_color = "#666666"
            facecolor = "none"
            lw = 1.0
            alpha = 0.9

        if facecolor != "none":
            ax.fill(
                poly_ax_x, poly_ax_y, facecolor=facecolor, alpha=alpha,
                edgecolor="none", zorder=4,
            )
        ax.plot(
            poly_ax_x, poly_ax_y, color=edge_color, linewidth=lw + 0.6,
            zorder=5,
        )
        ax.plot(
            poly_ax_x, poly_ax_y, color=base_color, linewidth=lw,
            alpha=alpha if facecolor == "none" else 0.95, zorder=5,
        )

        front_len = box["length"] * 0.4
        ax.annotate(
            "",
            xy=(box["cy"] + sin_h * front_len, box["cx"] + cos_h * front_len),
            xytext=(box["cy"], box["cx"]),
            arrowprops=dict(arrowstyle="->", color=base_color, lw=lw * 0.8),
        )

    _draw_ego(ax, size=2.5)

    for d in range(-40, 50, 10):
        ax.axhline(d, color="#dddddd", lw=0.4, alpha=0.8)
        ax.axvline(d, color="#dddddd", lw=0.4, alpha=0.8)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)

    legend_items = [
        mpatches.Patch(facecolor=_OSZ_PALETTE["road"], label="Road"),
        mpatches.Patch(facecolor=_OSZ_PALETTE["grass"], label="Non-drivable ground"),
        mpatches.Patch(facecolor=_OSZ_PALETTE["obstacle"], label="Occluder (LiDAR/depth)"),
        mpatches.Patch(
            facecolor=_OSZ_PALETTE["osz"],
            label="PA-relevant OSZ ({}) cells".format(osz_pa.sum()),
        ),
        mpatches.Patch(
            facecolor="#ff9800", edgecolor="#e65100", alpha=0.55,
            label="Vehicle (visible) — can cause OSZ",
        ),
        plt.Line2D(
            [0], [0], color="#ff0000", lw=2,
            label="Vehicle in OSZ ({}) — phantom candidate".format(n_phantom),
        ),
        mpatches.Patch(facecolor="#AAAAAA", label="Other object"),
    ]
    ax.legend(handles=legend_items, fontsize=7, loc="upper right", framealpha=0.9)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(
        "y (m)  ← ego-left | ego-right →", fontsize=9,
        color=_OSZ_PALETTE["text"],
    )
    ax.set_ylabel("x (m)  ↑ forward", fontsize=9, color=_OSZ_PALETTE["text"])
    ax.tick_params(labelsize=8, colors=_OSZ_PALETTE["text"])
    ax.set_title(
        "BEV GT + PA-relevant OSZ  |  {}...\n"
        "{} annotations  |  {} phantom candidates".format(
            sample_token[:16], len(boxes), n_phantom),
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        print("[saved] {}".format(save_path))

    return fig


def plot_osz_explained(
    osz_pa: np.ndarray,
    bev_occ: np.ndarray,
    drivable_mask: Optional[np.ndarray] = None,
    bev_range: Tuple[float, float, float, float] = BEV_RANGE_M,
    sample_token: str = "",
    title_extra: str = "",
    save_path: Optional[str] = None,
    draw_lanes: bool = False,
    nusc=None,
) -> plt.Figure:
    """绘制单面板 BEV 解释图：遮挡物、OSZ 阴影、道路与自车。

    Parameters
    ----------
    osz_pa : np.ndarray
        OSZ 阴影掩码 ``(nx, ny)``。
    bev_occ : np.ndarray
        BEV 占据 / 遮挡物掩码 ``(nx, ny)``。
    drivable_mask : np.ndarray 或 None, optional
        可行驶区域掩码；``None`` 时背景退化为"占据之外皆道路"。
    bev_range : Tuple[float, float, float, float], optional
        BEV 范围 ``(x_min, x_max, y_min, y_max)``，单位米。
    sample_token : str, optional
        图标题中显示的样本 token。
    title_extra : str, optional
        追加到标题的额外文本。
    save_path : str 或 None, optional
        给定时保存 PNG 到该路径。
    draw_lanes : bool, optional
        是否叠加 HD 地图车道线；需要 ``nusc`` 与 ``sample_token``。
    nusc : NuScenes 或 None, optional
        ``draw_lanes`` 为 True 时用于车道渲染的 nuScenes 实例。

    Returns
    -------
    matplotlib.figure.Figure
        生成的图；调用方负责关闭（或改用 :func:`save_osz_explained`）。
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    extent, xlim, ylim = bev_extent(bev_range)
    nx, ny = osz_pa.shape

    overlay = np.zeros((nx, ny, 3), dtype=np.float32)
    if drivable_mask is not None and drivable_mask.any():
        non_drivable = ~(drivable_mask | bev_occ)
        overlay[non_drivable] = _OSZ_PALETTE["grass"]
        overlay[drivable_mask] = _OSZ_PALETTE["road"]
    else:
        overlay[:] = _OSZ_PALETTE["grass"]
        overlay[~bev_occ] = _OSZ_PALETTE["road"]

    overlay[bev_occ] = _OSZ_PALETTE["obstacle"]
    if drivable_mask is not None and drivable_mask.any():
        overlay[osz_pa & drivable_mask] = _OSZ_PALETTE["osz"]
    else:
        overlay[osz_pa] = _OSZ_PALETTE["osz"]

    ax.imshow(overlay, origin="lower", extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    _draw_ego(ax, size=2.5)

    # 可选 HD 地图车道线（装饰层，失败静默跳过）。
    if draw_lanes and nusc is not None and sample_token:
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
            import pyquaternion

            location = get_map_name(nusc, sample_token)
            lidar_sd_token = nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
            ep = ego_pose_from_sample_data(nusc, lidar_sd_token)
            ego_t = np.array(ep["translation"], dtype=np.float64)
            ego_q = pyquaternion.Quaternion(ep["rotation"])
            if location:
                nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=location)
                recs = nusc_map.get_records_in_radius(
                    float(ego_t[0]), float(ego_t[1]), 55.0,
                    ["lane", "road_segment"],
                )
                for layer in ("lane", "road_segment"):
                    for tok in recs.get(layer, []):
                        rec = nusc_map.get(layer, tok)
                        poly_rec = nusc_map.get("polygon", rec["polygon_token"])
                        nodes = [nusc_map.get("node", nt)
                                 for nt in poly_rec["exterior_node_tokens"]]
                        pts = []
                        for nd in nodes:
                            gpos = np.array(
                                [nd["x"], nd["y"], 0.0], dtype=np.float32)
                            delta = gpos - ego_t
                            epos = ego_q.inverse.rotate(delta)
                            pts.append((epos[1], epos[0]))
                        if len(pts) >= 2:
                            xs, ys = zip(*pts)
                            ax.plot(
                                xs, ys, "-", color=_OSZ_PALETTE["lane"],
                                linewidth=0.4, alpha=0.5, zorder=1,
                            )
        except Exception:
            pass

    for d in range(-40, 50, 10):
        ax.axhline(d, color="#666666", lw=0.3, alpha=0.35)
        ax.axvline(d, color="#666666", lw=0.3, alpha=0.35)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.axvline(0, color="#888888", lw=0.6)

    n_osz = int(osz_pa.sum())
    n_occ = int(bev_occ.sum())
    pct_osz = n_osz / max(osz_pa.size, 1) * 100
    pct_occ = n_occ / max(bev_occ.size, 1) * 100
    title = (
        "OSZ explained | {:.1f}% OSZ ({} cells) "
        "| {:.1f}% occluders ({} cells)".format(pct_osz, n_osz, pct_occ, n_occ)
    )
    if sample_token:
        title = "{}\n{}...".format(title, sample_token[:24])
    if title_extra:
        title = "{}  {}".format(title, title_extra)
    ax.set_title(
        title, fontsize=10, fontweight="bold", color=_OSZ_PALETTE["text"])

    ax.set_xlabel(
        "y (m)  ← ego-left | ego-right →", fontsize=8,
        color=_OSZ_PALETTE["text_mid"],
    )
    ax.set_ylabel(
        "x (m)  ↑ forward", fontsize=8, color=_OSZ_PALETTE["text_mid"]
    )
    ax.tick_params(labelsize=7)

    handles = [
        mpatches.Patch(color=_OSZ_PALETTE["road"], label="Road (drivable)"),
        mpatches.Patch(color=_OSZ_PALETTE["grass"], label="Non-drivable ground"),
        mpatches.Patch(
            color=_OSZ_PALETTE["obstacle"],
            label="Occluders ({} cells) — cause the shadow".format(n_occ),
        ),
        mpatches.Patch(
            color=_OSZ_PALETTE["osz"],
            label="OSZ ({} cells) — the shadow".format(n_osz),
        ),
        plt.Line2D(
            [0], [0], marker="^", color=_OSZ_PALETTE["ego"],
            markersize=8, linestyle="none", label="Ego",
        ),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        print("[saved] {}".format(save_path))

    return fig


def save_gt_osz(
    osz_pa: np.ndarray,
    bev_occ: np.ndarray,
    drivable_mask: Optional[np.ndarray],
    nusc,
    sample_token: str,
    save_path: str,
    bev_range: Tuple[float, float, float, float] = BEV_RANGE_M,
    bev_res: float = BEV_RESOLUTION_M,
) -> None:
    """渲染并保存 GT 叠加图，随后关闭 figure。

    参数含义与 :func:`plot_gt_osz` 相同；``save_path`` 为必填。
    """
    fig = plot_gt_osz(
        osz_pa, bev_occ, drivable_mask, nusc, sample_token,
        bev_range=bev_range, bev_res=bev_res, save_path=save_path,
    )
    plt.close(fig)


def save_osz_explained(
    osz_pa: np.ndarray,
    bev_occ: np.ndarray,
    drivable_mask: Optional[np.ndarray] = None,
    bev_range: Tuple[float, float, float, float] = BEV_RANGE_M,
    sample_token: str = "",
    title_extra: str = "",
    save_path: Optional[str] = None,
    draw_lanes: bool = False,
    nusc=None,
) -> None:
    """渲染并保存 OSZ 解释图，随后关闭 figure。

    参数含义与 :func:`plot_osz_explained` 相同。
    """
    fig = plot_osz_explained(
        osz_pa, bev_occ, drivable_mask=drivable_mask,
        bev_range=bev_range, sample_token=sample_token,
        title_extra=title_extra, save_path=save_path,
        draw_lanes=draw_lanes, nusc=nusc,
    )
    plt.close(fig)
