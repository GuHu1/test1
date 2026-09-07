"""OSZ 挖掘管线的全局配置。

本文件包含两类高度内聚的配置：

1. 管线全局常量：高度阈值、相机列表、MiDaS 路径、可行驶区域图层等。
2. BEV 网格定义：网格范围、分辨率、尺寸计算与 ``BEVGrid`` 运行时类。

原则：不要让多个文件各自硬编码格子尺寸，然后指望没人只改其中两处。
下面只有一个旋钮。要改就改这里；所有从此文件导入的模块自动生效。

布局约定（nuScenes ego 系，x = 前向，y = 左向）：

* BEV 数组形状 ``(nx, ny)``，``indexing='ij'``：
  轴 0 为 ego-x（后 -> 前，递增），轴 1 为 ego-y（左 -> 右，**递减**）。
* ``BEV_RANGE_XYXY`` 存 ``(x_min, x_max, y_min, y_max)``，单位米。

兼容性：Python 3.8 / numpy 1.19.5 / torch 1.9.1+cu111。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 项目布局（权重 / 第三方模型代码位于 osz 包内部）
# ---------------------------------------------------------------------------
_OSZ_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# BEV 网格几何
# ---------------------------------------------------------------------------
#: 以 ego 为中心的半边长（米），x、y 方向相同。
BEV_EXTENT_M = 50.0

#: 每个 BEV 格子的边长（米）。这是控制全项目网格分辨率的唯一数值。
BEV_RESOLUTION_M = 0.2

#: ``(x_min, x_max, y_min, y_max)``，单位米，以 ego 为中心。
BEV_RANGE_XYXY = (
    -BEV_EXTENT_M,
    BEV_EXTENT_M,
    -BEV_EXTENT_M,
    BEV_EXTENT_M,
)

#: 兼容历史 OSZ 配置命名的别名。
BEV_RANGE_M = BEV_RANGE_XYXY

_nx = int(round((BEV_RANGE_XYXY[1] - BEV_RANGE_XYXY[0]) / BEV_RESOLUTION_M))
_ny = int(round((BEV_RANGE_XYXY[3] - BEV_RANGE_XYXY[2]) / BEV_RESOLUTION_M))
assert _nx == _ny, (
    "BEV 网格必须是正方形 —— "
    f"得到 ({_nx},{_ny})。除非上面的 BEV_EXTENT_M/BEV_RESOLUTION_M "
    "被改得不一致，否则不应出现这种情况。"
)

#: ego-x（前向）方向的格子数。
BEV_NX = _nx

#: ego-y（左向）方向的格子数。
BEV_NY = _ny


# ---------------------------------------------------------------------------
# 障碍物 / 自由空间分类的高度阈值
# ---------------------------------------------------------------------------
Z_MIN_M: float = 0.8
"""判定为障碍物的最小高度（米）。低于此值视为地面。"""

Z_MAX_M: float = 3.0
"""障碍物的最大可信高度（米）；超过此值的点被过滤（建筑/天空）。"""

OBSERVER_HEIGHT_M: float = 1.2
"""眼高射线投射所用虚拟观察者的眼高（米）。"""

EGO_CLEARANCE_RADIUS_M: float = 1.0
"""自车中心周围的清空半径（米），用于消除自遮挡伪影。"""

AGE_DT_S: float = 0.5
"""盲区年龄递推默认步长（秒），对应 nuScenes 关键帧间隔。"""

# ---------------------------------------------------------------------------
# 相机与深度
# ---------------------------------------------------------------------------
NUSCENES_CAMERAS: List[str] = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
"""nuScenes 标准相机名。"""

MAX_METRIC_DEPTH_M: float = 70.0
"""度量深度截断上限（米），超过则视为无效以避免远处幻影。"""

MIDAS_MODEL_PATH: str = str(_OSZ_ROOT / "weights" / "midas_v21_small_256.pt")
"""本地 MiDaS v2.1 Small checkpoint 路径；运行时绝不联网下载。"""

MIDAS_REPO_PATH: str = str(_OSZ_ROOT / "third_party" / "MiDaS")
"""包含 ``hubconf.py`` 的本地 MiDaS 仓库路径。"""

MIDAS_MODEL_URL: str = (
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/"
    "midas_v21_small_256.pt"
)
"""服务器手动部署 checkpoint 时参考的上游下载地址。"""

MIN_ALIGN_POINTS: int = 20
"""LiDAR 尺度对齐所需的最少有效点数。"""

# ---------------------------------------------------------------------------
# 可行驶区域过滤
# ---------------------------------------------------------------------------
DRIVABLE_MAP_LAYERS: List[str] = ["drivable_area", "carpark_area"]
"""构成可行驶区域的 HD 地图图层。"""

EXCLUDED_MAP_LAYERS: List[str] = ["walkway", "ped_crossing"]
"""从可行驶掩码中显式剔除的 HD 地图图层。"""

DEFAULT_DRIVABLE_DILATION_M: float = 1.5
"""可行驶掩码的形态学膨胀半径（米）。"""


# ---------------------------------------------------------------------------
# BEV 网格工具函数与运行时类
# ---------------------------------------------------------------------------
def describe() -> str:
    """返回当前 BEV 网格的可读摘要。

    Returns
    -------
    str
        网格尺寸、分辨率、范围与总格子数。
    """
    return (
        f"BEV grid: {BEV_NX}x{BEV_NY} cells @ {BEV_RESOLUTION_M}m/cell "
        f"(±{BEV_EXTENT_M}m range, {BEV_NX * BEV_NY:,} cells total)"
    )


def bev_grid_shape(
    bev_range: Tuple[float, float, float, float],
    bev_res: float,
) -> Tuple[int, int]:
    """由 ``(x_min, x_max, y_min, y_max)`` 范围计算 ``(nx, ny)``。

    使用 ``round`` 取整，避免非整数比值带来的浮点残差悄悄把网格
    缩小一格。

    Parameters
    ----------
    bev_range : Tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)``，单位米。
    bev_res : float
        格子边长（米）。

    Returns
    -------
    Tuple[int, int]
        ``(nx, ny)`` 网格尺寸。
    """
    x_min, x_max, y_min, y_max = bev_range
    nx = int(round((x_max - x_min) / bev_res))
    ny = int(round((y_max - y_min) / bev_res))
    return nx, ny


def bev_extent(
    bev_range: Tuple[float, float, float, float]
) -> Tuple[list, Tuple[float, float], Tuple[float, float]]:
    """返回 BEV 范围对应的 matplotlib ``extent``、``xlim`` 与 ``ylim``。

    Parameters
    ----------
    bev_range : Tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)``，单位米。

    Returns
    -------
    extent : list
        ``[y_max, y_min, x_min, x_max]`` —— 横轴为 ego-y（反转，使
        ego 左侧显示在图左侧），纵轴为 ego-x（前向朝上）。
    xlim : Tuple[float, float]
        ``(y_max, y_min)``，供 ``ax.set_xlim()``。
    ylim : Tuple[float, float]
        ``(x_min, x_max)``，供 ``ax.set_ylim()``。
    """
    x_min, x_max, y_min, y_max = bev_range
    return [y_max, y_min, x_min, x_max], (y_max, y_min), (x_min, x_max)


class BEVGrid:
    """运行时 BEV 网格定义（范围 + 分辨率 + 尺寸）。

    现行高度感知管线中的所有几何计算（BEV 高度图构建、射线投射）
    都以同一个 ``BEVGrid`` 实例作为网格唯一来源。该类只描述二维
    网格几何，不携带障碍物高度阈值 —— 高度门控属于逐调用的语义
    参数，默认值见本模块常量。

    Parameters
    ----------
    bev_range : Tuple[float, float, float, float], optional
        ``(x_min, x_max, y_min, y_max)``，单位米。默认为模块常量
        ``BEV_RANGE_XYXY``（±50 m）。
    bev_res : float, optional
        格子边长（米）。默认为模块常量 ``BEV_RESOLUTION_M``（0.2 m）。

    Attributes
    ----------
    bev_range : Tuple[float, float, float, float]
        网格范围。
    bev_res : float
        格子边长（米）。
    nx, ny : int
        ego-x / ego-y 方向的格子数。

    Examples
    --------
    >>> grid = BEVGrid()      # 使用模块默认值：±50 m @ 0.2 m
    >>> grid.nx, grid.ny
    (500, 500)
    """

    def __init__(
        self,
        bev_range: Tuple[float, float, float, float] = BEV_RANGE_XYXY,
        bev_res: float = BEV_RESOLUTION_M,
    ):
        self.bev_range = tuple(float(v) for v in bev_range)
        self.bev_res = float(bev_res)
        self.nx, self.ny = bev_grid_shape(self.bev_range, self.bev_res)

    def __repr__(self) -> str:
        return (
            f"BEVGrid(nx={self.nx}, ny={self.ny}, res={self.bev_res}m, "
            f"range={self.bev_range})"
        )


if __name__ == '__main__':
    grid = BEVGrid()
    print(describe())
    print(f"  BEV_RANGE_XYXY = {BEV_RANGE_XYXY}")
    print(f"  {grid!r}")
