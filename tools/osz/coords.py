"""坐标变换与 ego 系工具
=====================

包内坐标相关查询的唯一实现处。

原则：不要把同一段取数逻辑复制粘贴到多个文件里然后指望它们保持
同步。这里保留的每个函数都有至少一个现行调用方。

速查（nuScenes ego 系）
----------------------
* ``x`` = 前向，``y`` = 左向，``z`` = 上向。
* BEV 数组形状 ``(nx, ny)``，``indexing='ij'``：
  ``i``（轴 0）= ego-x，``j``（轴 1）= ego-y。

兼容性：Python 3.8 / numpy 1.19.5 / nuscenes-devkit 1.1.9。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def get_map_name(nusc, sample_token: str) -> str:
    """返回样本所在的地图名（如 ``'boston-seaport'``）。

    Parameters
    ----------
    nusc : NuScenes
        nuScenes 数据集句柄。
    sample_token : str
        样本 token。

    Returns
    -------
    str
        地图位置名。
    """
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    return log["location"]


def ego_pose_from_sample_data(nusc, sample_data_token: str) -> dict:
    """返回 sample_data 记录对应的 ego_pose 原始记录。

    调用方自行从记录中取 ``translation`` / ``rotation``，保留完整的
    旋转信息（四元数或旋转矩阵由调用方决定）。

    Parameters
    ----------
    nusc : NuScenes
        nuScenes 数据集句柄。
    sample_data_token : str
        sample_data 记录的 token。

    Returns
    -------
    dict
        nuScenes ego_pose 原始记录。
    """
    sd = nusc.get("sample_data", sample_data_token)
    return nusc.get("ego_pose", sd["ego_pose_token"])

# ---------------------------------------------------------------------------
# BEV 框几何与 GT 查询（GT 叠加可视化使用）
# ---------------------------------------------------------------------------
def bev_box_corners_ego(
    x_ego: float, y_ego: float, heading: float, w: float, l: float
) -> np.ndarray:
    """返回 ego 系下矩形框的四个 BEV 角点。

    ``heading=0`` 表示框朝向 ``+x``（前向）；正值航向为左转
    （逆时针，朝 ``+y``）。``w`` 为宽（横向），``l`` 为长（前向）。

    Parameters
    ----------
    x_ego : float
        框中心的 ego 系 x（米）。
    y_ego : float
        框中心的 ego 系 y（米）。
    heading : float
        框在 ego 系下的航向角（弧度）。
    w : float
        框宽（米）。
    l : float
        框长（米）。

    Returns
    -------
    np.ndarray
        ``(4, 2)`` 角点数组，每行为 ``(x_ego, y_ego)``。
    """
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    local = np.array(
        [[l / 2, w / 2],
         [l / 2, -w / 2],
         [-l / 2, -w / 2],
         [-l / 2, w / 2]],
        dtype=np.float32,
    )
    corners = local @ R.T
    corners[:, 0] += x_ego
    corners[:, 1] += y_ego
    return corners


def get_vehicle_states_ego(
    nusc, sample_token: str, bev_range: Tuple[float, float, float, float]
) -> List[dict]:
    """返回 BEV 范围内的 ego 系标注框状态。

    Parameters
    ----------
    nusc : NuScenes
        nuScenes 数据集句柄。
    sample_token : str
        样本 token。
    bev_range : Tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)``，单位米。

    Returns
    -------
    List[dict]
        每个字典含 ``cx, cy, length, width, yaw, category, token,
        in_osz``；``in_osz`` 由调用方（GT 叠加可视化）填充。
    """
    # 延迟导入：保持本模块在无 pyquaternion 的环境下仍可被导入
    # （例如纯 mock 冒烟），pyquaternion 缺失只影响本函数调用方。
    import pyquaternion

    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    ego_t = np.array(ep["translation"], dtype=np.float64)
    ego_q = pyquaternion.Quaternion(ep["rotation"])

    x_min, x_max, y_min, y_max = bev_range
    boxes = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        delta = np.array(ann["translation"]) - ego_t
        pt_ego = ego_q.inverse.rotate(delta)

        cx, cy = float(pt_ego[0]), float(pt_ego[1])
        if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
            continue

        box_q = pyquaternion.Quaternion(ann["rotation"])
        box_q_ego = ego_q.inverse * box_q
        yaw_ego = box_q_ego.yaw_pitch_roll[0]

        boxes.append({
            "cx": cx,
            "cy": cy,
            "length": ann["size"][1],
            "width": ann["size"][0],
            "yaw": yaw_ego,
            "category": ann["category_name"],
            "token": ann_token,
            "in_osz": False,
        })
    return boxes


#: 车辆类别集合（GT 叠加可视化用于区分车辆与其他物体）。
VEHICLE_CATEGORIES = {
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "vehicle.trailer",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
}
