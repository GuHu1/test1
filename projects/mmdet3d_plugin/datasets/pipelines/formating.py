
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D

@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """
    def __init__(self, class_names, with_gt=True, with_label=True, with_ego=True):
        super(CustomDefaultFormatBundle3D, self).__init__(class_names, with_gt, with_label)
        self.with_ego = with_ego


    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        # results['gt_map_masks'] = DC(to_tensor(results['gt_map_masks']), stack=True)
        if self.with_ego:
            if 'ego_his_trajs' in results:
                results['ego_his_trajs'] = DC(to_tensor(results['ego_his_trajs'][None, ...]), stack=True)
            if 'ego_fut_trajs' in results:
                results['ego_fut_trajs'] = DC(to_tensor(results['ego_fut_trajs'][None, ...]), stack=True)
            if 'ego_fut_masks' in results:
                results['ego_fut_masks'] = DC(to_tensor(results['ego_fut_masks'][None, None, ...]), stack=True)
            if 'ego_fut_cmd' in results:
                results['ego_fut_cmd'] = DC(to_tensor(results['ego_fut_cmd'][None, None, ...]), stack=True)
            if 'ego_lcf_feat' in results:
                results['ego_lcf_feat'] = DC(to_tensor(results['ego_lcf_feat'][None, None, ...]), stack=True)
            if 'gt_attr_labels' in results:
                results['gt_attr_labels'] = DC(to_tensor(results['gt_attr_labels']), cpu_only=False)
            # OSZ 资产: (F, H, W) float32, 与 ego 相关键保持相同的 None-包装约定
            if 'osz_vis' in results:
                results['osz_vis'] = DC(to_tensor(results['osz_vis'][None, ...]), stack=True)
            if 'osz_age' in results:
                results['osz_age'] = DC(to_tensor(results['osz_age'][None, ...]), stack=True)
            if 'osz_vis_next' in results:
                results['osz_vis_next'] = DC(to_tensor(results['osz_vis_next'][None, ...]), stack=True)
            if 'osz_vis_next_valid' in results:
                results['osz_vis_next_valid'] = DC(to_tensor(np.array(
                    [[results['osz_vis_next_valid']]], dtype=np.float32)), stack=True)

        return results