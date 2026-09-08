# ResWorld + OSZ (Occlusion Shadow Zone) 配置
# 在基线 resworld_config.py 之上:
#   1. 数据管线注入 LoadOSZAnnotations (当前+相邻帧 OSZ 资产加载、对齐、重采样)
#   2. head 开启遮挡特定处理: 残差解耦 + 永续门控 + 风险场注入 + 注意力偏置
# 前置: 先运行 tools/osz/run_export.py 生成 data/osz/npz/{token}.npz
_base_ = ['./resworld_config.py']

# OSZ 资产根目录 (tools/osz/run_export.py 的 --outdir/npz)
osz_root = 'data/osz/npz'

# head BEV 网格 (与 base 配置 grid_config / bev_h_ / bev_w_ 一致)
_osz_bev_size = (100, 100)
_osz_pc_range = (-15.0, 15.0, -30.0, 30.0)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config={{ _base_.data_config }},
        load_point_label=True,
        sequential=True),
    dict(
        type='LoadOSZAnnotations',
        bev_size=_osz_bev_size,
        pc_range=_osz_pc_range,
        allow_missing=False),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range={{ _base_.point_cloud_range }}),
    dict(type='CustomObjectNameFilter', classes={{ _base_.class_names }}),
    dict(type='CustomDefaultFormatBundle3D', class_names={{ _base_.class_names }}, with_ego=True),
    dict(type='CustomCollect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img_inputs', 'ego_his_trajs', 'gt_depth', 'can_bus',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels',
               'osz_vis', 'osz_age'])
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=False,
        data_config={{ _base_.data_config }},
        load_point_label=False,
        sequential=True),
    dict(
        type='LoadOSZAnnotations',
        bev_size=_osz_bev_size,
        pc_range=_osz_pc_range,
        allow_missing=False),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range={{ _base_.point_cloud_range }}),
    dict(type='CustomObjectNameFilter', classes={{ _base_.class_names }}),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='CustomDefaultFormatBundle3D', class_names={{ _base_.class_names }},
                 with_label=False, with_ego=True),
            dict(type='CustomCollect3D',
                 keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'fut_valid_flag', 'can_bus',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                       'ego_lcf_feat', 'gt_attr_labels',
                       'osz_vis', 'osz_age', 'osz_vis_next', 'osz_vis_next_valid'])])
]

data = dict(
    train=dict(osz_root=osz_root, pipeline=train_pipeline),
    val=dict(osz_root=osz_root, pipeline=test_pipeline),
    test=dict(osz_root=osz_root, pipeline=test_pipeline),
)

# DDP 容错: OSZ 各子模块按样本/消融组合条件性参与前向 (如 osz_vis_trans_head
# 在部分样本的 loss 中可能缺席), 开启后 reducer 不再因"参数未收到梯度"中断训练
find_unused_parameters = True

model = dict(
    pts_bbox_head=dict(
        # --- OSZ 遮挡特定处理开关 ---
        use_osz=True,            # 总开关
        osz_permanence=True,     # IDEA4: 永续门控, 聚合 R_t 与被丢弃的 R_{t-1}
        osz_disentangle=True,    # IDEA1a: 残差按可见性解耦为 运动/涌现 双 token 流
        osz_risk_inject=True,    # IDEA3: 盲区年龄风险场注入 B_future 合成
        osz_attn_bias=2.0,       # IDEA2a: TokenLearner 注意力 logit 偏置强度 (0=关)
        osz_age_tau=2.0,         # 盲区年龄归一化时间常数(秒)
        osz_occ_tokens=4,        # IDEA1b: 幻影token数, 从高龄盲区提取 (0=关)
        osz_occ_bias=4.0,        # IDEA1b: 幻影token的风险logit偏置强度
        osz_vis_trans_weight=1.0,  # IDEA1c: 可见性转移自监督BCE损失权重 (0=关)
        osz_attn_cover_weight=0.1, # IDEA2b: 注意力覆盖(分布交叉熵)损失权重 (0=关)
    ))
