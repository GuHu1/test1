import copy
from math import pi, cos, sin
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import BaseModule
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn import Linear, bias_init_with_prob
from torch.cuda.amp.autocast_mode import autocast
from mmdet.models.backbones.resnet import BasicBlock

from .tokenlearner import *
from .rcsample import Mlp, SELayer
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

class MLN(nn.Module):
    ''' 
    from "https://github.com/exiawsh/StreamPETR"
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out

class SELayerMLP(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)

@HEADS.register_module()
class ResWorldHead(BaseModule):
    def __init__(self,
                #  *args,
                 grid_config,
                 num_frames=3,
                 embed_dims=256,
                 in_channels=256,
                 num_reg_fcs=2,
                 positional_encoding=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 num_scenes=16,
                 latent_decoder=None,
                 res_latent_decoder=None,
                 way_decoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 use_osz=False,
                 osz_permanence=False,
                 osz_disentangle=False,
                 osz_risk_inject=False,
                 osz_attn_bias=0.0,
                 osz_age_tau=2.0,
                 osz_occ_tokens=0,
                 osz_occ_bias=4.0,
                 osz_vis_trans_weight=0.0,
                 osz_attn_cover_weight=0.0,
                 **kwargs):
        super(ResWorldHead, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.latent_decoder = latent_decoder
        self.res_latent_decoder = res_latent_decoder
        self.way_decoder = way_decoder
        self.positional_encoding = positional_encoding
        self.ego_fut_mode = ego_fut_mode
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_scenes = num_scenes
        self.num_frames = num_frames
        # OSZ (遮挡阴影区) 相关开关
        self.use_osz = use_osz                  # 总开关
        self.osz_permanence = osz_permanence    # 永续门控: 加权聚合多步残差
        self.osz_disentangle = osz_disentangle  # 残差按可见性解耦为 运动/涌现
        self.osz_risk_inject = osz_risk_inject  # 盲区年龄风险场注入 B_future
        self.osz_attn_bias = osz_attn_bias      # TokenLearner 注意力 logit 偏置强度 (0=关)
        self.osz_age_tau = osz_age_tau          # 盲区年龄归一化时间常数(秒)
        self.grid_min = torch.tensor([grid_config['x'][0], grid_config['y'][0]])
        self.grid_max = torch.tensor([grid_config['x'][1], grid_config['y'][1]])
        self.grid_size = torch.tensor([grid_config['x'][2], grid_config['y'][2]])

        self._init_layers()
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_reg_init = build_loss(loss_plan_reg)
                
    def _init_layers(self):
        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        init_ego_fut_decoder = []
        for _ in range(self.num_reg_fcs):
            init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            init_ego_fut_decoder.append(nn.ReLU())
        init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.init_ego_fut_decoder = nn.Sequential(*init_ego_fut_decoder)

        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        self.navi_se = SELayerMLP(self.embed_dims)
        self.canbus_mlp = Mlp(18, self.embed_dims, self.embed_dims)
        self.canbus_se = SELayerMLP(self.embed_dims)
        self.bev_fusion_conv = ConvModule(self.in_channels * self.num_frames, self.in_channels, 
                                          kernel_size=3, padding=1)

        self.way_point = nn.Embedding(self.ego_fut_mode*self.fut_ts, self.embed_dims * 2)
        self.tokenlearner = TokenLearner(self.num_scenes, self.embed_dims * 2)
        self.res_tokenlearner = TokenLearner(self.num_scenes, self.embed_dims * 2)
        # 残差解耦后 token 数翻倍 (运动 token + 涌现 token), 幻影 token 追加在后
        tf_num_tokens = self.num_scenes * (2 if self.osz_disentangle else 1) \
            + (self.osz_occ_tokens if self.use_osz else 0)
        self.tokenfuser = TokenFuser(tf_num_tokens, 256)
        if self.osz_risk_inject:
            # 风险场注入: 以几何风险图为空间门控, 让 B_future 在盲区内携带风险特征
            self.osz_risk_conv = nn.Conv2d(self.in_channels, self.in_channels,
                                           kernel_size=3, padding=1)
            self.osz_risk_beta = nn.Parameter(torch.zeros(1))  # 零初始化, 保证恒等起步
        if self.use_osz and self.osz_occ_tokens > 0:
            # 幻影 token: 从高龄盲区提取的查询, 代表可能存在的隐藏物体
            self.occ_tokenlearner = TokenLearner(self.osz_occ_tokens, self.embed_dims * 2)
        if self.use_osz and self.osz_vis_trans_weight > 0:
            # 可见性转移预测头: 从 B_future 预测下一帧可见性图 (自监督, 免人工标注)
            self.osz_vis_trans_head = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channels // 2, 1, kernel_size=1))
        else:
            self.osz_vis_trans_head = None

        self.latent_decoder = build_transformer_layer_sequence(self.latent_decoder)
        self.way_decoder = build_transformer_layer_sequence(self.way_decoder)
        self.res_latent_decoder = build_transformer_layer_sequence(self.res_latent_decoder)
        self.col_attn = MultiScaleDeformableAttention(self.embed_dims,
                                            num_points=8, num_levels=1)
        self.action_mln = MLN(6*2)
        self.positional_encoding = build_positional_encoding(
            self.positional_encoding)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        if self.latent_decoder is not None:
            for p in self.latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 
        if self.way_decoder is not None:
            for p in self.way_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.res_latent_decoder is not None:
            for p in self.res_latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 

    @force_fp32(apply_to=('bev_feats'))
    def forward(self,
                bev_inputs,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                cmd=None,
                osz_vis=None,
                osz_age=None
            ):

        bev_feats, can_bus_infos = bev_inputs
        bt, c, h, w = bev_feats.shape
        bs = bt // self.num_frames
        # OSZ: 统一为 (bs, num_frames, H, W)
        if osz_vis is not None and osz_vis.dim() == 5:
            osz_vis = osz_vis.squeeze(1)
        if osz_age is not None and osz_age.dim() == 5:
            osz_age = osz_age.squeeze(1)
        if self.use_osz:
            if osz_vis is None and (self.osz_disentangle or self.osz_permanence
                                    or self.osz_risk_inject or self.osz_attn_bias > 0):
                raise RuntimeError(
                    'use_osz=True 但未收到 osz_vis, 请检查: '
                    '1) 数据集 osz_root 配置; 2) pipeline 含 LoadOSZAnnotations; '
                    '3) Collect keys 含 osz_vis/osz_age; 4) tools/osz 资产已导出')
            if osz_vis is not None and osz_age is None:
                osz_age = torch.zeros_like(osz_vis)
        dtype = bev_feats[0].dtype
        device = bev_feats[0].device
        can_bus_infos = self.canbus_mlp(can_bus_infos.permute(1, 0, 2)).view(bt, 1, self.in_channels)
        bev_embed = self.canbus_se(bev_feats.view(bt, c, h*w).permute(0, 2, 1), can_bus_infos)
        bev_embed_single = bev_embed.clone()
        bev_embed = bev_embed.permute(0, 2, 1).view(self.num_frames, bs, c, h, w).permute(1, 0, 2, 3, 4)
        bev_embed = self.bev_fusion_conv(bev_embed.reshape(bs, self.num_frames * c, h, w))
        bev_feats = bev_feats.view(self.num_frames, bs, c, h, w)


        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_feats.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)
        bev_embed = bev_embed.reshape(bs, c, h * w).permute(0, 2, 1)
        # res_embed = res_embed.reshape(bs, c, h * w).permute(0, 2, 1)

        navi_embed = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            navi_embed.append(self.navi_embedding.weight[cmd_idx][None, None])
        navi_embed = torch.cat(navi_embed, dim=0)

        bev_navi_embed = self.navi_se(bev_embed, navi_embed)

        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)

        # --- OSZ: 当前帧风险图 (高龄盲区), 供注意力偏置/幻影token/风险注入复用 ---
        osz_vis_flat = None
        risk0 = None
        if self.use_osz and osz_vis is not None:
            assert osz_vis.shape[0] == bs and osz_vis.shape[1] == self.num_frames, \
                'osz_vis shape {} mismatch (bs={}, num_frames={})'.format(
                    tuple(osz_vis.shape), bs, self.num_frames)
            osz_vis_flat = osz_vis.reshape(bs, self.num_frames, -1).to(dtype)
            osz_age0 = osz_age[:, 0].to(dtype)
            risk0 = (1.0 - osz_vis[:, 0].to(dtype)) * \
                    (osz_age0 / (osz_age0 + self.osz_age_tau))      # (bs,H,W)
        attn_bias = None
        if risk0 is not None and self.osz_attn_bias > 0:
            attn_bias = (self.osz_attn_bias * risk0).reshape(bs, 1, -1)

        learned_latent_query, selected = self.tokenlearner(bev_query)
        _, res_selected = self.res_tokenlearner(bev_query, logit_bias=attn_bias)

        # --- OSZ: 幻影 token, 从高龄盲区提取 (代表可能存在的隐藏物体) ---
        occ_tokens = None
        if risk0 is not None and self.osz_occ_tokens > 0:
            occ_bias = (self.osz_occ_bias * risk0).reshape(bs, 1, -1)
            occ_tokens, _ = self.occ_tokenlearner(bev_query, logit_bias=occ_bias)  # (bs,N_occ,2C)

        # --- OSZ: 每查询逐帧可见度 (res_selected 的空间支撑与可见性图收缩) ---
        vis_q = None
        if osz_vis_flat is not None:
            # (bs, Ns, HW) x (bs, F, HW) -> (bs, F, Ns)
            vis_q = torch.einsum('bsn,bfn->bfs', res_selected, osz_vis_flat).clamp(0., 1.)
        bev_embed_single = torch.cat((bev_embed_single, pos_embd.repeat(self.num_frames,1,1)), -1)
        bev_embed_single = torch.einsum('bsi,bic->bsc', res_selected.repeat(self.num_frames,1,1), bev_embed_single)\
                            .view(self.num_frames, bs, learned_latent_query.shape[1],learned_latent_query.shape[2])
        res_latent_query_all = bev_embed_single[:-1] - bev_embed_single[1:]

        learned_latent_query=learned_latent_query.permute(1, 0, 2)
        latent_query, latent_pos = torch.split(
            learned_latent_query, self.embed_dims, dim=2)

        latent_query = self.latent_decoder(
                query=latent_query,
                key=latent_query,
                value=latent_query,
                query_pos=latent_pos,
                key_pos=latent_pos)

        way_point = self.way_point.weight.to(dtype)
        wp_pos, way_point = torch.split(
            way_point, self.embed_dims, dim=1)

        wp_pos = wp_pos.unsqueeze(0).expand(bs, -1, -1)
        way_point = way_point.unsqueeze(0).expand(bs, -1, -1)
        wp_pos = wp_pos.permute(1, 0, 2)
        way_point = way_point.permute(1, 0, 2)

        way_point = self.way_decoder(
                query=way_point,
                key=latent_query,
                value=latent_query,
                query_pos=wp_pos,
                key_pos=latent_pos)
        init_ego_trajs = self.init_ego_fut_decoder(way_point)
        init_ego_trajs = init_ego_trajs.permute(1, 0, 2). view(bs, 
                                                    self.ego_fut_mode, self.fut_ts, 2)
        init_ego_coords = init_ego_trajs.cumsum(dim=2).view(bs, -1, 2)
        init_ego_coords = (init_ego_coords - self.grid_min.to(device).view(1, 1, 2)) / \
                            self.grid_size.to(device).view(1, 1, 2) / 200
        init_wp_vector = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            init_wp_vector.append(init_ego_trajs[bidx, cmd_idx, ...].reshape(1, 1, 12))  
        init_wp_vector = torch.cat(init_wp_vector, dim=1)

        reference_points = init_ego_coords.unsqueeze(2)
        spatial_shapes = torch.tensor([self.bev_w, self.bev_h]).view(1, 2).to(device)
        level_start_index = torch.tensor([0]).to(device)

        # --- OSZ: 残差可见性解耦 + 永续门控聚合 ---
        # res_latent_query_all: (F-1, bs, Ns, 2C), [0]=S_t-S_{t-1}, [1]=S_{t-1}-S_{t-2}
        res_tokens = res_latent_query_all[0]
        if vis_q is not None and (self.osz_disentangle or self.osz_permanence):
            # 残差 i 的联合可见度 (相邻两帧均可见 => 真运动)
            psi = torch.sqrt(vis_q[:, :-1] * vis_q[:, 1:]).unsqueeze(-1)  # (bs,F-1,Ns,1)
            R_all = res_latent_query_all.permute(1, 0, 2, 3)               # (bs,F-1,Ns,2C)
            if self.osz_disentangle:
                R_motion = R_all * psi
                R_emerge = R_all * (1.0 - psi)
            else:
                R_motion = R_all
            if self.osz_permanence and R_all.shape[1] > 1:
                # 永续门控: 当前可见 -> 信任最新残差; 刚被遮挡 -> 回退到旧残差 (匀速外推先验)
                g0 = vis_q[:, 0]                                          # (bs,Ns)
                g_rest = (1.0 - vis_q[:, 0:1]) * vis_q[:, 1:]             # (bs,F-2,Ns)
                gates = torch.cat([g0.unsqueeze(1), g_rest], dim=1)       # (bs,F-1,Ns)
                gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)
                Rm = (R_motion * gates.unsqueeze(-1)).sum(dim=1)          # (bs,Ns,2C)
            else:
                Rm = R_motion[:, 0]
            if self.osz_disentangle:
                # 涌现 token 只取最近一次可见性跳变
                res_tokens = torch.cat([Rm, R_emerge[:, 0]], dim=1)       # (bs,2Ns,2C)
            else:
                res_tokens = Rm

        # 幻影 token 追加在残差 token 之后, 共同参与 TR-World 自注意力
        if occ_tokens is not None:
            res_tokens = torch.cat([res_tokens, occ_tokens], dim=1)

        res_latent_query=res_tokens.permute(1, 0, 2)
        res_latent_query, res_latent_pos = torch.split(
            res_latent_query, self.embed_dims, dim=2)
        res_latent_query = self.action_mln(res_latent_query, init_wp_vector)
        res_latent_query = self.res_latent_decoder(
                query=res_latent_query,
                key=res_latent_query,
                value=res_latent_query,
                query_pos=res_latent_pos,
                key_pos=res_latent_pos)
        
        pred_bev = self.tokenfuser(res_latent_query.permute(1, 0, 2), bev_navi_embed) + bev_navi_embed

        # --- OSZ: 盲区年龄风险场注入 B_future 合成 ---
        # 高龄盲区携带风险特征, FGTR 在参考点处可 attend 到"可能有物体涌现"的区域
        if self.use_osz and self.osz_risk_inject and osz_vis is not None:
            osz_age0 = osz_age[:, 0].to(dtype)
            risk_map = (1.0 - osz_vis[:, 0].to(dtype)) * \
                       (osz_age0 / (osz_age0 + self.osz_age_tau))      # (bs,H,W)
            risk_map = risk_map.unsqueeze(1)                            # (bs,1,H,W)
            pred_bev_2d = pred_bev.permute(0, 2, 1).reshape(bs, c, h, w)
            pred_bev_2d = pred_bev_2d + self.osz_risk_beta * risk_map * \
                          self.osz_risk_conv(pred_bev_2d)
            pred_bev = pred_bev_2d.reshape(bs, c, h * w).permute(0, 2, 1)

        way_point = self.col_attn(
                query=way_point,
                key=pred_bev.permute(1, 0, 2),
                value=pred_bev.permute(1, 0, 2),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index) 

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        outputs_ego_trajs = outputs_ego_trajs.permute(1, 0, 2). view(bs, 
                                                      self.ego_fut_mode, self.fut_ts, 2)

        wp_vector = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            wp_vector.append(outputs_ego_trajs[bidx, cmd_idx, ...].reshape(1, 1, 12))  
        wp_vector = torch.cat(wp_vector, dim=1)

        outs = {
            'bev_embed': bev_embed,
            'pred_bev': pred_bev,
            'scene_query': latent_query,
            'wp_vector': wp_vector,
            # 'act_query': act_query,
            # 'act_pos': act_pos,
            'ego_fut_preds': outputs_ego_trajs,
            # 'ego_fut_preds': init_ego_trajs,
            'init_ego_fut_preds': init_ego_trajs,
            'osz_vis_q': vis_q,  # (bs,F,Ns) 每查询逐帧可见度, 供分析/调试
            'res_selected': res_selected,      # (bs,Ns,HW) 残差提取注意力, 覆盖损失用
            'vis_trans_pred': vis_trans_pred,  # (bs,1,H,W) 下一帧可见性预测 logits
        }

        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None,
             osz_vis=None,
             osz_age=None,
             osz_vis_next=None,
             osz_vis_next_valid=None):

        ego_fut_preds = preds_dicts['ego_fut_preds']
        init_ego_fut_preds = preds_dicts['init_ego_fut_preds']
        loss_dict = dict()

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        loss_plan_l1_init = self.loss_plan_reg_init(
            init_ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )
      
        loss_dict['loss_plan_reg'] = loss_plan_l1
        loss_dict['loss_plan_reg_init'] = loss_plan_l1_init

        # --- OSZ: 可见性转移自监督损失 (IDEA1) ---
        # 用 B_future 预测下一帧可见性图; 标签来自 OSZ 资产, 免人工标注;
        # 刻意只监督 1 通道可见性而非稠密 BEV 外观 (避免 Table 4 中稠密监督的负面影响)
        if self.use_osz and self.osz_vis_trans_weight > 0 and \
                osz_vis_next is not None and preds_dicts.get('vis_trans_pred') is not None:
            vis_pred = preds_dicts['vis_trans_pred']              # (bs,1,H,W) logits
            # collate 后为 (bs,1,H,W), 直接 reshape 对齐预测
            tgt = osz_vis_next.reshape(vis_pred.shape).to(vis_pred.dtype)
            valid = osz_vis_next_valid.reshape(-1).to(vis_pred.dtype) if \
                osz_vis_next_valid is not None else torch.ones(
                    vis_pred.shape[0], dtype=vis_pred.dtype, device=vis_pred.device)
            l_trans = F.binary_cross_entropy_with_logits(
                vis_pred, tgt, reduction='none').mean(dim=(1, 2, 3))   # (bs,)
            loss_dict['loss_osz_vis_trans'] = self.osz_vis_trans_weight * \
                (l_trans * valid).sum() / (valid.sum() + 1e-6)

        # --- OSZ: 注意力覆盖损失 (IDEA2) ---
        # 风险图与注意力质量分布的交叉熵: 让残差提取注意力向高龄盲区倾斜
        if self.use_osz and self.osz_attn_cover_weight > 0 and osz_vis is not None:
            if osz_age is None:
                osz_age = torch.zeros_like(osz_vis)
            if osz_vis.dim() == 5:
                osz_vis = osz_vis.squeeze(1)
            if osz_age.dim() == 5:
                osz_age = osz_age.squeeze(1)
            res_sel = preds_dicts['res_selected']                 # (bs,Ns,HW)
            bs_l = res_sel.shape[0]
            osz_age0 = osz_age[:, 0].reshape(bs_l, -1).to(res_sel.dtype)
            risk_flat = ((1.0 - osz_vis[:, 0].reshape(bs_l, -1).to(res_sel.dtype)) *
                         (osz_age0 / (osz_age0 + self.osz_age_tau)))   # (bs,HW)
            risk_sum = risk_flat.sum(-1, keepdim=True)
            valid_c = (risk_sum.squeeze(-1) > 1e-6).to(res_sel.dtype)  # 无遮挡样本不计
            risk_norm = risk_flat / (risk_sum + 1e-6)
            attn_mass = res_sel.mean(dim=1)                            # (bs,HW)
            attn_norm = attn_mass / (attn_mass.sum(-1, keepdim=True) + 1e-6)
            l_cover = -(risk_norm * (attn_norm + 1e-9).log()).sum(-1)  # (bs,)
            loss_dict['loss_osz_attn_cover'] = self.osz_attn_cover_weight * \
                (l_cover * valid_c).sum() / (valid_c.sum() + 1e-6)

        return loss_dict


