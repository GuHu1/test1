"""OSZ 批量导出：OSZ npz 资产、盲区年龄与逐帧可视化。

逐帧管线
--------
1. 加载 nuScenes 样本（六相机图像、LiDAR 扫描、标定）。
2. 每相机只推理一次度量深度（本地 MiDaS v2.1 Small + LiDAR 对齐）；
   同一张深度图同时供 OSZ 计算与逐相机可视化消费。
3. 将深度图反投影到 ego 系并构建 BEV 高度图。
4. 运行高度感知 360° 射线投射 -> ground OSZ 与 eye OSZ。
5. 沿场景时间轴递推累加盲区年龄。
6. 可选：将 OSZ 与 nuScenes HD 地图可行驶区域求交。
7. 导出 ``npz/{token}.npz``、逐帧可视化与 summary CSV。

后端
----
本移植版仅保留 NumPy 几何后端（原 Torch/CUDA 后端未随包迁移）；
``--backend`` 只接受 ``numpy``。深度估计仍可用 torch（MiDaS 推理），
由 ``--device`` 控制。

``--outdir`` 下的输出布局::

    npz/{token}.npz            OSZ 掩码 + 盲区年龄 + 元数据
    viz/{token}/{CAM}.jpg      原始六视角相机图
    viz/{token}/{CAM}_osz.png  逐相机 OSZ 深度叠加图
    viz/{token}/osz_bev.png    BEV 六面板 OSZ 汇总图
    viz/{token}/gt_osz.png     GT 框 + OSZ 叠加图（仅真实数据）
    viz/{token}/osz_explained.png  OSZ 解释图（仅真实数据）
    summary.csv                逐帧格子统计

Examples
--------
从 ResWorld 仓库根目录运行。真实 nuScenes 导出::

    python tools/osz/run_export.py \\
        --dataroot /data/sets/nuscenes \\
        --version v1.0-trainval \\
        --outdir data/osz \\
        --scene-shard 0 --num-scene-shards 8

按 token 处理单样本::

    python tools/osz/run_export.py --dataroot /data/sets/nuscenes \\
        --sample_token <TOKEN>

合成 mock（无需 nuScenes / MiDaS）::

    python tools/osz/run_export.py --mock --outdir ./osz_output

兼容性：Python 3.8 / numpy 1.19.5 / torch 1.9.1+cu111 /
matplotlib 3.5.3 / scipy 1.10.1 / nuscenes-devkit 1.1.9。
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sys.path 引导：本脚本作为脚本从仓库根执行
# （``python tools/osz/run_export.py``），此时 sys.path[0] 是
# ``tools/osz`` 而非仓库根。参照 tools/create_data.py 的
# ``sys.path.append('.')`` 模式，这里按文件位置把仓库根（``tools``
# 的父目录）加入 sys.path，使 ``tools.osz`` 包可被绝对导入，
# 与当前工作目录无关。
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.osz.osz_config import (  # noqa: E402
    AGE_DT_S,
    BEV_RANGE_M,
    BEV_RESOLUTION_M,
    OBSERVER_HEIGHT_M,
    Z_MAX_M,
    Z_MIN_M,
)
from tools.osz.osz_config import BEVGrid, bev_grid_shape  # noqa: E402
from tools.osz.depth_estimator import (  # noqa: E402
    DepthEstimator,
    MockDepthEstimator,
)
from tools.osz.drivable_filter import build_drivable_mask  # noqa: E402
from tools.osz.loader import NuScenesOSZLoader  # noqa: E402
from tools.osz.pipeline import (  # noqa: E402
    apply_drivable_mask,
    compute_osz,
    prepare_working_depth,
    update_occlusion_age,
)
from tools.osz.viz import (  # noqa: E402
    save_gt_osz,
    save_osz_explained,
    save_osz_panels,
    visualize_frame_cameras,
)


def _scene_tokens(nusc, scene):
    """按时间顺序产出场景的全部样本 token。"""
    token = scene['first_sample_token']
    while token:
        sample = nusc.get('sample', token)
        yield token
        token = sample['next']


def _build_estimator(source, device, mock_only=False):
    """按深度来源创建深度估计器。

    ``mock_only`` 强制使用 LiDAR 补全兜底估计器（合成 mock 数据
    场景）。
    """
    if mock_only or source == 'lidar':
        return MockDepthEstimator()
    estimator = DepthEstimator(device=device)
    estimator.load()
    return estimator


def _drivable_for_token(loader, token, mock, bev_range, bev_res):
    """返回 token 的可行驶区域掩码；未启用时返回 None。"""
    if not mock:
        return build_drivable_mask(loader.nusc, token, bev_range, bev_res)
    nx, ny = bev_grid_shape(bev_range, bev_res)
    print("[INFO] mock 模式：可行驶掩码为全 True。")
    return np.ones((nx, ny), dtype=bool)


def _export_frame(
    frame,
    working_cameras,
    depth_used,
    grid,
    estimator,
    args,
    age,
    loader=None,
):
    """计算、落盘并可视化单帧。

    Parameters
    ----------
    frame : dict
        loader 产出的原始帧（可视化输入）。
    working_cameras : dict
        已填好 ``depth_used`` 的相机字典。
    depth_used : dict
        相机名 -> 实际使用的深度图。
    grid : BEVGrid
        定义 BEV 网格的实例。
    estimator : object
        深度估计器（已应用；保留以维持 API 对称）。
    args : argparse.Namespace
        已解析的 CLI 参数。
    age : np.ndarray
        来自场景上一帧的输入盲区年龄。
    loader : NuScenesOSZLoader, optional
        真实数据模式下提供 ``nusc`` 句柄以构建可行驶掩码。

    Returns
    -------
    dict
        summary CSV 的一行；外发 ``age`` 数组放在 ``age`` 键下。
    """
    token = frame['sample_token']
    height, ground, eye = compute_osz(
        working_cameras, grid,
        observer_height=args.observer_height,
        use_uncertainty=args.use_uncertainty)
    semi = ground & ~eye
    age = update_occlusion_age(age, eye, args.dt)

    drivable = None
    if args.use_drivable:
        drivable = _drivable_for_token(
            loader, token, args.mock, BEV_RANGE_M, BEV_RESOLUTION_M)
        ground, eye, semi, age = apply_drivable_mask(
            ground, eye, semi, age, drivable)

    out_root = Path(args.outdir)
    npz_path = out_root / 'npz' / f'{token}.npz'
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        osz_eye=eye.astype(np.uint8),
        osz_ground=ground.astype(np.uint8),
        semi=semi.astype(np.uint8),
        occlusion_age=age,
        bev_height=height.astype(np.float32),
        bev_range=np.asarray(BEV_RANGE_M, dtype=np.float32),
        bev_resolution=np.asarray(BEV_RESOLUTION_M, dtype=np.float32),
        axis_order=np.asarray('x_inc_y_desc'),
        age_dt=np.asarray(args.dt, dtype=np.float32),
    )
    if drivable is not None:
        arrays['drivable'] = drivable.astype(np.uint8)
    np.savez_compressed(npz_path, **arrays)
    print(f"[saved] {npz_path}")

    if not args.no_viz:
        viz_dir = out_root / 'viz' / token
        visualize_frame_cameras(
            frame, depth_used, viz_dir,
            z_min=Z_MIN_M, z_max=Z_MAX_M)
        save_osz_panels(
            height, ground, eye, semi, age, grid,
            args.observer_height, token,
            viz_dir / 'osz_bev.png', drivable_mask=drivable)

        # GT 叠加与 OSZ 解释图：仅真实 nuScenes 数据时输出。
        if (not args.mock and loader is not None
                and getattr(loader, 'nusc', None) is not None):
            bev_occ = height > 0.05
            try:
                save_gt_osz(
                    ground, bev_occ, drivable, loader.nusc, token,
                    save_path=str(viz_dir / 'gt_osz.png'))
                save_osz_explained(
                    ground, bev_occ, drivable,
                    sample_token=token,
                    save_path=str(viz_dir / 'osz_explained.png'),
                    draw_lanes=True, nusc=loader.nusc)
            except Exception as e:
                print(f'[WARN] GT 叠加可视化失败：{e}')

    total = height.size
    bev_occ = height > 0.05
    row = {
        'sample_token': token,
        'occupied': int(bev_occ.sum()),
        'osz_ground': int(ground.sum()),
        'osz_eye': int(eye.sum()),
        'semi': int(semi.sum()),
        'age_max': float(age.max()) if age.size else 0.0,
        'age_mean': float(age[eye].mean()) if eye.any() else 0.0,
        'osz_ground_ratio': float(ground.sum()) / max(total, 1),
        'osz_eye_ratio': float(eye.sum()) / max(total, 1),
        'age': age,
    }
    print(f"[frame] {token}: ground={row['osz_ground']} eye={row['osz_eye']} "
          f"semi={row['semi']} age_max={row['age_max']:.1f}s")
    return row


def _resume_age(npz_path, age, dt):
    """从既有 npz 资产恢复盲区年龄状态。"""
    with np.load(npz_path, allow_pickle=False) as old:
        if 'occlusion_age' in old:
            return old['occlusion_age'].astype(np.float32)
        if 'osz_eye' in old:
            old_eye = old['osz_eye'].astype(bool)
            return np.where(old_eye, age + dt, 0.0).astype(np.float32)
    return age


def _iter_real_frames(loader, args):
    """按场景顺序产出真实数据集的样本 token。

    npz 已存在且未加 ``--overwrite`` 时，该帧会被调用方跳过，
    但盲区年龄状态仍会继续推进。
    """
    nusc = loader.nusc
    scenes = list(nusc.scene)
    if args.info_pkl:
        with open(args.info_pkl, 'rb') as handle:
            packed = pickle.load(handle)
        infos = packed['infos'] if isinstance(packed, dict) \
            and 'infos' in packed else packed
        allowed_scenes = set(str(info['scene_token']) for info in infos)
        scenes = [scene for scene in scenes
                  if str(scene['token']) in allowed_scenes]
    scenes = scenes[args.scene_shard::args.num_scene_shards]
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    for scene in scenes:
        for token in _scene_tokens(nusc, scene):
            yield token


def build_args():
    """构建导出器的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="OSZ export: OSZ npz + occlusion age + visualizations "
                    "(NumPy backend)"
    )
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='nuScenes 数据根目录。')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='nuScenes 数据集版本。')
    parser.add_argument('--outdir', type=str, default='./osz_output',
                        help='输出根目录。')
    parser.add_argument('--sample-token', '--sample_token',
                        dest='sample_token', type=str, default=None,
                        help='只处理单个样本（按 token）。')
    parser.add_argument('--mock', action='store_true',
                        help='使用合成 mock 数据（无需 nuScenes）。')
    parser.add_argument('--depth-source', choices=('midas', 'lidar'),
                        default='midas',
                        help='相机深度来源：本地 MiDaS 或 LiDAR 补全投影。')
    parser.add_argument('--backend', choices=('numpy',),
                        default='numpy',
                        help="OSZ 几何后端。本移植版仅保留 NumPy 后端"
                             "（原 torch/CUDA 后端未迁移）。")
    parser.add_argument('--device', default=None,
                        help='MiDaS 深度估计的 torch 设备（如 cuda / cpu）；'
                             '默认按 CUDA 可用性自动选择。')
    parser.add_argument('--n-sweeps', type=int, default=0,
                        help='聚合的历史 LiDAR 扫描数。')
    parser.add_argument('--dt', type=float, default=AGE_DT_S,
                        help='盲区年龄递推步长（秒）。')
    parser.add_argument('--observer-height', type=float,
                        default=OBSERVER_HEIGHT_M,
                        help='观察者眼高（米）。')
    parser.add_argument('--info-pkl', default=None,
                        help='可选的 info pkl，用于限定场景集合。')
    parser.add_argument('--scene-shard', type=int, default=0,
                        help='并行导出用的场景分片索引。')
    parser.add_argument('--num-scene-shards', type=int, default=1,
                        help='场景分片总数。')
    parser.add_argument('--max-scenes', type=int, default=0,
                        help='处理这么多场景后停止（0 = 全部）。')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='写入这么多样本后停止（0 = 全部）。')
    parser.add_argument('--overwrite', action='store_true',
                        help='重算已存在的 npz 资产。')
    parser.add_argument('--use-drivable', action='store_true',
                        help='将 OSZ 与可行驶区域掩码求交。')
    parser.add_argument('--use-uncertainty', action='store_true',
                        help='逆不确定性相机/LiDAR 融合。')
    parser.add_argument('--no-viz', action='store_true',
                        help='跳过全部可视化输出。')
    return parser


def main():
    """从命令行运行 OSZ 导出。"""
    args = build_args().parse_args()
    if not 0 <= args.scene_shard < args.num_scene_shards:
        raise ValueError('scene-shard 必须落在 [0, num-scene-shards) 内')

    # NumPy 后端固定，无需解析；--device 仅控制 MiDaS 推理设备。
    if args.device and args.device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                f'--device {args.device} 被指定，但当前环境无可用 CUDA')

    loader = NuScenesOSZLoader(
        dataroot=args.dataroot, version=args.version,
        max_samples=0, n_sweeps=args.n_sweeps, force_mock=args.mock)
    if not args.mock and loader.use_mock:
        raise RuntimeError(
            '在 {} 未找到 nuScenes 数据'.format(args.dataroot))
    if args.mock and args.sample_token:
        raise ValueError('mock 模式下不支持 --sample_token')

    estimator = _build_estimator(
        args.depth_source, args.device, mock_only=args.mock)
    grid = BEVGrid()

    out_root = Path(args.outdir)
    rows = []
    written = 0
    age = np.zeros((grid.nx, grid.ny), dtype=np.float32)

    def _handle_frame(frame):
        nonlocal written, age
        working = prepare_working_depth(frame['cameras'], estimator)
        depth_used = {name: cam['depth_used']
                      for name, cam in working.items()}
        row = _export_frame(
            frame, working, depth_used, grid, estimator, args, age,
            loader=loader)
        age = row.pop('age')
        rows.append(row)
        written += 1

    if args.mock:
        for frame in loader:
            _handle_frame(frame)
            if args.max_samples and written >= args.max_samples:
                break
    elif args.sample_token:
        _handle_frame(loader.build_frame_for_token(args.sample_token))
    else:
        for token in _iter_real_frames(loader, args):
            npz_path = out_root / 'npz' / f'{token}.npz'
            if npz_path.exists() and not args.overwrite:
                # 场景下一帧的年龄仍需继续推进。
                age = _resume_age(npz_path, age, args.dt)
                continue
            _handle_frame(loader.build_frame_for_token(token))
            if args.max_samples and written >= args.max_samples:
                break

    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        suffix = ''
        if args.n_sweeps:
            suffix += f'_sweeps{args.n_sweeps}'
        if args.use_drivable:
            suffix += '_drivable'
        if args.mock:
            suffix += '_mock'
        csv_path = out_root / f'summary{suffix}.csv'
        fieldnames = [
            'sample_token', 'occupied', 'osz_ground', 'osz_eye', 'semi',
            'age_max', 'age_mean', 'osz_ground_ratio', 'osz_eye_ratio',
        ]
        with open(csv_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {csv_path}")

    print('OSZ EXPORT DONE: {} samples'.format(written))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
