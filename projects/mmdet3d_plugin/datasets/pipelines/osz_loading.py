# OSZ (Occlusion Shadow Zone) 资产加载与对齐
# 将 tools/osz 离线生产的逐帧 OSZ npz 资产接入训练/测试 pipeline：
#   1. 读取当前帧与相邻帧的 osz_eye / occlusion_age
#   2. 通过 ego2global 位姿把各帧 OSZ 从各自自车系 warp 到当前帧自车系
#      （与 BEV 特征的对齐方式一致：所有帧的 BEV 都构建在当前 key-ego 系）
#   3. 裁剪+重采样到 head 级 BEV 网格（默认 100x100, x∈[-15,15], y∈[-30,30]）
# 输出:
#   results['osz_vis']: (F, H, W) float32, 1=可见, 0=被眼高以上障碍物遮挡
#   results['osz_age']: (F, H, W) float32, 盲区年龄(秒)
import os

import numpy as np
from scipy.ndimage import map_coordinates

from mmdet.datasets.builder import PIPELINES

# OSZ npz 原生网格约定（与 tools/osz 生产端一致，见 OSZMiner README）
OSZ_X_MIN, OSZ_X_MAX = -50.0, 50.0
OSZ_Y_MIN, OSZ_Y_MAX = -50.0, 50.0
OSZ_RES = 0.2  # 米/格
OSZ_GRID = 500  # 500x500
# axis_order = 'x_inc_y_desc': 行(axis0)随 ego-x 递增, 列(axis1)随 ego-y 递减


def _yaw_from_quaternion_wxyz(q):
    """从 [w, x, y, z] 四元数提取偏航角（z 轴）。"""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _sample_osz(src, px, py, cval):
    """在 OSZ 网格上按 ego 坐标(px, py) 双线性采样。

    Args:
        src: (500, 500) 源图, 行=x递增, 列=y递减
        px, py: (H, W) 源帧 ego 系坐标(米)
        cval: 越界填充值
    """
    rows = (px - OSZ_X_MIN) / OSZ_RES - 0.5
    cols = (OSZ_Y_MAX - py) / OSZ_RES - 0.5
    return map_coordinates(src, [rows, cols], order=1, mode='constant', cval=cval)


@PIPELINES.register_module()
class LoadOSZAnnotations(object):
    """加载并对齐 OSZ 资产到当前帧 BEV 网格。

    Args:
        bev_size (tuple): 输出 BEV 网格 (H, W), 与 head 的 bev_h/bev_w 一致
        pc_range (tuple): (x_min, x_max, y_min, y_max), 与 grid_config 的 x/y 一致
        allow_missing (bool): npz 缺失时是否回退为全可见(vis=1, age=0);
            False 则直接报错, 防止训练时静默丢失 OSZ 监督
        dump_dir (str | None): 若设置, 将前 dump_max 个样本的重采样结果落盘供人工核对
        dump_max (int): dump_dir 下最多落盘的样本数
    """

    def __init__(self,
                 bev_size=(100, 100),
                 pc_range=(-15.0, 15.0, -30.0, 30.0),
                 allow_missing=False,
                 dump_dir=None,
                 dump_max=20):
        self.bev_h, self.bev_w = bev_size
        self.x_min, self.x_max, self.y_min, self.y_max = pc_range
        self.allow_missing = allow_missing
        self.dump_dir = dump_dir
        self.dump_max = dump_max
        self._dump_count = 0

        # 目标 BEV 网格(当前帧 ego 系)的格心坐标
        xs = self.x_min + (np.arange(self.bev_w) + 0.5) * \
            (self.x_max - self.x_min) / self.bev_w
        ys = self.y_min + (np.arange(self.bev_h) + 0.5) * \
            (self.y_max - self.y_min) / self.bev_h
        # gx 对应列(x 维), gy 对应行(y 维), 与 head 中 (h, w) 展平顺序一致
        self._gx, self._gy = np.meshgrid(xs, ys)

    def _load_npz(self, path):
        """读取一帧 OSZ 资产, 返回 (vis_src, age_src) 原生 500x500 网格。"""
        if not os.path.exists(path):
            if not self.allow_missing:
                raise FileNotFoundError(
                    'OSZ asset not found: {}. Run tools/osz/run_export.py '
                    'first, or set allow_missing=True for debugging.'.format(path))
            return (np.ones((OSZ_GRID, OSZ_GRID), dtype=np.float32),
                    np.zeros((OSZ_GRID, OSZ_GRID), dtype=np.float32))
        data = np.load(path)
        osz_eye = data['osz_eye'].astype(np.float32)
        age = data['occlusion_age'].astype(np.float32)
        vis_src = 1.0 - osz_eye
        return vis_src, np.clip(age, 0.0, None)

    def _warp_resample(self, vis_src, age_src, yaw_src, t_src, yaw_tgt, t_tgt):
        """把源帧 OSZ warp+重采样到目标(当前)帧 BEV 网格。

        对每个目标格心 p_tgt: p_global = R_tgt @ p_tgt + t_tgt,
        p_src = R_src^-1 @ (p_global - t_src), 再在源网格双线性采样。
        """
        ct, st = np.cos(yaw_tgt), np.sin(yaw_tgt)
        x_g = ct * self._gx - st * self._gy + t_tgt[0]
        y_g = st * self._gx + ct * self._gy + t_tgt[1]
        dx = x_g - t_src[0]
        dy = y_g - t_src[1]
        cs, ss = np.cos(yaw_src), np.sin(yaw_src)
        x_s = cs * dx + ss * dy
        y_s = -ss * dx + cs * dy
        vis = _sample_osz(vis_src, x_s, y_s, cval=1.0).astype(np.float32)
        age = _sample_osz(age_src, x_s, y_s, cval=0.0).astype(np.float32)
        return vis, age

    def __call__(self, results):
        if 'osz_path' not in results.get('curr', {}):
            # 数据集未配置 osz_root, 直接跳过 (兼容原 pipeline)
            return results

        frames = [results['curr']] + list(results.get('adjacent', []))
        # 目标(当前)帧位姿
        tgt = frames[0]
        yaw_tgt = _yaw_from_quaternion_wxyz(tgt['ego2global_rotation'])
        t_tgt = np.array(tgt['ego2global_translation'][:2], dtype=np.float64)

        vis_list, age_list = [], []
        for i, finfo in enumerate(frames):
            vis_src, age_src = self._load_npz(finfo['osz_path'])
            if i == 0:
                yaw_src, t_src = yaw_tgt, t_tgt
            else:
                yaw_src = _yaw_from_quaternion_wxyz(finfo['ego2global_rotation'])
                t_src = np.array(finfo['ego2global_translation'][:2], dtype=np.float64)
            vis, age = self._warp_resample(vis_src, age_src,
                                           yaw_src, t_src, yaw_tgt, t_tgt)
            vis_list.append(vis)
            age_list.append(age)

        results['osz_vis'] = np.stack(vis_list, axis=0)  # (F, H, W)
        results['osz_age'] = np.stack(age_list, axis=0)  # (F, H, W)

        # 下一帧可见性 (可见性转移自监督目标): 场景末尾写零并置 valid=0
        next_info = results.get('next_info', None)
        if next_info is not None and 'osz_path' in next_info:
            vis_n_src, age_n_src = self._load_npz(next_info['osz_path'])
            yaw_n = _yaw_from_quaternion_wxyz(next_info['ego2global_rotation'])
            t_n = np.array(next_info['ego2global_translation'][:2], dtype=np.float64)
            vis_next, _ = self._warp_resample(vis_n_src, age_n_src,
                                              yaw_n, t_n, yaw_tgt, t_tgt)
            results['osz_vis_next'] = vis_next.astype(np.float32)
            results['osz_vis_next_valid'] = np.float32(1.0)
        else:
            results['osz_vis_next'] = np.zeros((self.bev_h, self.bev_w),
                                               dtype=np.float32)
            results['osz_vis_next_valid'] = np.float32(0.0)

        if self.dump_dir is not None and self._dump_count < self.dump_max:
            os.makedirs(self.dump_dir, exist_ok=True)
            token = str(tgt.get('token', 'sample{}'.format(self._dump_count)))
            np.savez(os.path.join(self.dump_dir, '{}_osz_bev.npz'.format(token)),
                     vis=results['osz_vis'], age=results['osz_age'])
            self._dump_count += 1

        return results
