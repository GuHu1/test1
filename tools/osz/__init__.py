"""OSZ（Occlusion Shadow Zone，遮挡阴影区）离线数据生产工具包。

移植自 OSZMiner：从 nuScenes 多相机影像 + LiDAR 挖掘高度感知遮挡
阴影区与盲区年龄（occlusion age），并导出逐帧 npz 资产。

几何后端仅保留 NumPy 实现；深度来源为本地 MiDaS v2.1 Small 单目
深度估计 + LiDAR 稀疏深度尺度对齐（可选 LiDAR 补全兜底）。

兼容性：Python 3.8 / numpy 1.19.5 / torch 1.9.1+cu111 /
scipy 1.10.1 / nuscenes-devkit 1.1.9。
"""
