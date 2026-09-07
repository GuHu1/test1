"""将几何 OSZ 与 HD 地图的车辆 plausible 区域求交。

坐标系说明
----------
nuScenes ``get_map_geom(patch_angle=0)`` 返回以 ego 为中心、但
仍按全局北向对齐的局部坐标多边形。

而我们的 BEV 网格使用 ego 中心坐标：

- 轴 0 = ego 前向（ego +x）
- 轴 1 = ego 左向（ego +y）

全局 -> ego 的旋转是绕原点旋转 ``-ego_yaw`` 的二维旋转。我们直接
对多边形顶点施加该旋转（精确、无图像插值），再在 ego 系内栅格化。
这避免了旋转栅格带来的轴方向混淆与插值伪影。

兼容性：Python 3.8 / numpy 1.19.5 / shapely 2.0.7 /
nuscenes-devkit 1.1.9 / Pillow 10.x / scipy 1.10.1。
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from .osz_config import (
    BEV_RANGE_M,
    BEV_RESOLUTION_M,
    DEFAULT_DRIVABLE_DILATION_M,
    DRIVABLE_MAP_LAYERS,
    EXCLUDED_MAP_LAYERS,
    bev_grid_shape,
)
from .coords import get_map_name

try:
    from nuscenes.map_expansion.map_api import NuScenesMap
    import pyquaternion
    from shapely import affinity
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False


_map_cache = {}


def get_nusc_map(dataroot: str, map_name: str):
    """返回缓存的 ``NuScenesMap`` 实例。

    Parameters
    ----------
    dataroot : str
        nuScenes 数据集根目录。
    map_name : str
        要加载的地图名（如 ``'singapore-onenorth'``）。

    Returns
    -------
    NuScenesMap
        缓存的地图实例。
    """
    key = (dataroot, map_name)
    if key not in _map_cache:
        try:
            _map_cache[key] = NuScenesMap(dataroot=dataroot, map_name=map_name)
        except Exception as e:
            raise RuntimeError(
                f"NuScenesMap({dataroot}, {map_name}) 加载失败：{e}  "
                f"—— {dataroot}/maps/ 下是否存在地图数据？"
            ) from e
    return _map_cache[key]


def _get_ego_pose(nusc, sample_token: str):
    """返回样本的 ego 位姿。

    Parameters
    ----------
    nusc : NuScenes
        nuScenes 数据集句柄。
    sample_token : str
        要查询的样本 token。

    Returns
    -------
    tuple
        ``(ego_translation_global, ego_yaw_rad)``。
    """
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    t = np.array(ep["translation"], dtype=np.float64)
    q = pyquaternion.Quaternion(ep["rotation"])
    return t, q.yaw_pitch_roll[0]


def _rotate_to_ego(geom, ego_yaw_rad: float):
    """把 Shapely 几何从全局对齐旋转到 ego 中心坐标。

    输入已以 ego 为中心（patch 中心），但仍按全局北向对齐。旋转
    ``-ego_yaw`` 使 ego 前向变为 +x。

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        待旋转的几何。
    ego_yaw_rad : float
        ego 偏航角（弧度）。

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        ego 中心坐标系下的旋转结果。
    """
    angle_deg = -np.degrees(ego_yaw_rad)
    return affinity.rotate(geom, angle_deg, origin=(0, 0), use_radians=False)


def _rasterize_polygons(
    geometries: list,
    bev_range,
    canvas_size,
    fill_value: int = 1,
) -> np.ndarray:
    """用 PIL 把 Shapely 几何栅格化为 (nx, ny) BEV 掩码。

    坐标映射（ego 中心）：

    - 度量 x ∈ [x_min, x_max] -> 像素列 ∈ [0, nx)
    - 度量 y ∈ [y_min, y_max] -> 像素行 ∈ [0, ny)

    Parameters
    ----------
    geometries : list
        待栅格化的 Shapely 几何列表。
    bev_range : tuple
        ``(x_min, x_max, y_min, y_max)``，单位米。
    canvas_size : tuple
        输出网格尺寸 ``(nx, ny)``。
    fill_value : int, optional
        多边形内部的填充值。

    Returns
    -------
    np.ndarray
        (nx, ny) 数组，``indexing='ij'``，与 ``BEVGrid`` 一致。
    """
    x_min, x_max, y_min, y_max = bev_range
    nx, ny = canvas_size
    scale_x = nx / (x_max - x_min)
    scale_y = ny / (y_max - y_min)

    img = Image.new("L", (nx, ny), 0)
    draw = ImageDraw.Draw(img)

    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        elif geom.geom_type == "GeometryCollection":
            polys = []
            for g in geom.geoms:
                if g.geom_type == "Polygon":
                    polys.append(g)
                elif g.geom_type == "MultiPolygon":
                    polys.extend(list(g.geoms))
        else:
            continue

        for poly in polys:
            if poly.is_empty:
                continue

            def _to_pix(coords):
                """把 (x, y) ego 坐标映射到 PIL 的 (col, row)。"""
                return [
                    ((x - x_min) * scale_x, (y_max - y) * scale_y)
                    for x, y in coords
                ]

            ext = _to_pix(poly.exterior.coords)
            if len(ext) >= 3:
                draw.polygon(ext, fill=fill_value)
            for interior in poly.interiors:
                hole = _to_pix(interior.coords)
                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)

    # PIL 图像为 (W=nx, H=ny)；np.array 得到 (H=ny, W=nx)。
    # 转置为 (nx, ny)，indexing='ij'。
    return np.array(img, dtype=np.uint8).T


def build_drivable_mask(
    nusc,
    sample_token: str,
    bev_range=BEV_RANGE_M,
    bev_res: float = BEV_RESOLUTION_M,
    dilation_m: float = DEFAULT_DRIVABLE_DILATION_M,
) -> np.ndarray:
    """构建 ego 中心 BEV 系下的 (nx, ny) bool 掩码。

    ``True`` 表示该格对车辆 plausible（道路/车道，非人行道）。
    HD 地图不可用时回退为全 ``True``。

    Parameters
    ----------
    nusc : NuScenes
        nuScenes 数据集句柄。
    sample_token : str
        要查询的样本 token。
    bev_range : tuple, optional
        ``(x_min, x_max, y_min, y_max)``，单位米。
    bev_res : float, optional
        BEV 格子边长（米）。
    dilation_m : float, optional
        形态学膨胀半径（米）。

    Returns
    -------
    np.ndarray
        (nx, ny) bool 可行驶区域掩码。
    """
    nx, ny = bev_grid_shape(bev_range, bev_res)

    if not MAP_AVAILABLE:
        return np.ones((nx, ny), dtype=bool)

    sample = nusc.get("sample", sample_token)
    map_name = get_map_name(nusc, sample_token)
    try:
        nusc_map = get_nusc_map(nusc.dataroot, map_name)
    except Exception as e:
        raise RuntimeError(
            f"NuScenesMap 加载失败：dataroot={nusc.dataroot}, map={map_name}: {e}"
        ) from e

    ego_t, ego_yaw = _get_ego_pose(nusc, sample_token)

    # patch_box 为全局坐标（米）；nuScenes API 使用 (x, y, H, W)。
    half_x = (bev_range[1] - bev_range[0]) / 2.0
    half_y = (bev_range[3] - bev_range[2]) / 2.0
    patch_box = (float(ego_t[0]), float(ego_t[1]), half_x * 2, half_y * 2)

    include_geoms = nusc_map.get_map_geom(
        patch_box, patch_angle=0.0, layer_names=DRIVABLE_MAP_LAYERS
    )
    exclude_geoms = nusc_map.get_map_geom(
        patch_box, patch_angle=0.0, layer_names=EXCLUDED_MAP_LAYERS
    )

    all_include, all_exclude = [], []
    for _, geom_list in include_geoms:
        all_include.extend(geom_list)
    for _, geom_list in exclude_geoms:
        all_exclude.extend(geom_list)

    include_ego = [_rotate_to_ego(g, ego_yaw) for g in all_include]
    exclude_ego = [_rotate_to_ego(g, ego_yaw) for g in all_exclude]

    canvas_size = (nx, ny)
    include_mask = _rasterize_polygons(include_ego, bev_range, canvas_size)
    exclude_mask = _rasterize_polygons(exclude_ego, bev_range, canvas_size)

    ego_mask = (include_mask > 0) & ~(exclude_mask > 0)

    if dilation_m > 0:
        dilation_px = max(1, int(round(dilation_m / bev_res)))
        ego_mask = binary_dilation(ego_mask, iterations=dilation_px)

    return ego_mask.astype(bool)
