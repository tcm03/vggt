# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from train_utils.general import check_and_fix_inf_nan
from math import ceil, floor

IGNORE_ORIGIN: float = -1000000.
REAL_EPS: float = 1e-6

@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        nonpad_masks = batch["nonpad_masks"]
        
        # Articulation type classification loss
        assert "cls_logits" in predictions, f"no cls_logits found, predictions keys: {predictions.keys()}"
        assert predictions["cls_logits"].ndim == 3, f"expected shape: (batch size, num frames, num labels)"
        assert batch["joint_types"].ndim == 2, f"expected shape: (batch size, num frames)"
        assert batch["joint_types"].max().item() <= predictions["cls_logits"].shape[2] - 1, f'incorrect labels: max label = {batch["joint_types"].max().item()}, but num labels = {predictions["cls_logits"].shape[2]}'
        cls_preds = predictions["cls_logits"].permute(0, 2, 1) # [2026-02-11] @tcm: bring channel dimension ahead
        gt_joint_types = batch["joint_types"]
        cls_loss: torch.Tensor = F.cross_entropy(
            cls_preds, 
            gt_joint_types, 
            reduction = "mean"
        )
        assert torch.isnan(cls_loss).any() == False, f"Found NaNs in cls_loss: {cls_loss}"
        total_loss += cls_loss
        loss_dict["loss_joint_type"] = cls_loss
        
        # Axis prediction loss: [0., 2.]
        assert "pose_enc" in predictions, f"no pose_enc found, predictions keys: {predictions.keys()}"
        joint_axis: torch.Tensor = predictions["pose_enc"][..., :3]
        assert joint_axis.shape == batch["joint_axes"].shape, f'unmatched shape: {joint_axis.shape} != {batch["joint_axes"].shape}'
        axis_norm_tensor: torch.Tensor = torch.linalg.vector_norm(joint_axis, dim=-1, keepdim=True) + REAL_EPS
        joint_axis = joint_axis / (axis_norm_tensor + REAL_EPS)
        gt_joint_axis = batch["joint_axes"]
        gt_axis_norm_tensor: torch.Tensor = torch.linalg.vector_norm(gt_joint_axis, dim=-1, keepdim=True) + REAL_EPS
        gt_joint_axis = gt_joint_axis / (gt_axis_norm_tensor + REAL_EPS)
        axis_loss: torch.Tensor = 1. - torch.einsum("...i,...i->...", joint_axis, gt_joint_axis) # [2026-02-11] @tcm: when pred joint axis is near 0, its normalized version is large => large loss
        assert axis_loss.ndim == 2, f"unexpected axis_loss.shape = {axis_loss.shape}"
        axis_loss = ((axis_loss * nonpad_masks).sum(dim=-1) / (nonpad_masks.sum(dim=-1) + REAL_EPS)).mean()
        assert torch.isnan(axis_loss).any() == False, f"Found NaNs in axis_loss: {axis_loss}"
        total_loss += axis_loss
        loss_dict["loss_joint_axis"] = axis_loss

        # Origin prediction loss
        joint_origin: torch.Tensor = predictions["pose_enc"][..., 3:]
        assert joint_origin.shape == batch["joint_origins"].shape, f'unmatched shape: {joint_origin.shape} != {batch["joint_origins"].shape}'
        gt_joint_origin: torch.Tensor = batch["joint_origins"]
        diff_norms = torch.linalg.vector_norm(joint_origin - gt_joint_origin, ord=2, dim=-1) + REAL_EPS
        origin_loss = (diff_norms * nonpad_masks * batch["revolute_origin_masks"])
        origin_loss = (origin_loss.sum(dim=-1) / (batch["revolute_origin_masks"].sum(dim=-1) + REAL_EPS)).mean()
        assert torch.isnan(origin_loss).any() == False, f"Found NaNs in origin_loss: {origin_loss}"
        total_loss += origin_loss
        loss_dict["loss_joint_origin"] = origin_loss

        assert "segmentation" in predictions, f"no segmentation found, predictions keys: {predictions.keys()}"
        # [2026-02-11] @tcm: predictions["segmentation"].shape = torch.Size([7, 256, 476, 518])
        # [2026-02-11] @tcm: batch["seg_images"].shape = [2, 7, 476, 518]
        assert predictions["segmentation"].ndim == 5 and predictions["segmentation"].shape[-1] == 3, f"expected shape: (B, S, H, W, 3)"
        assert batch["seg_images"].ndim == 4, f"expected shape: (B, S, H, W)"
        seg_pred = predictions["segmentation"].permute(0, 4, 1, 2, 3)
        seg_loss: torch.Tensor = F.cross_entropy(seg_pred, batch["seg_images"]) # [2026-02-06] @tcm: average over all loss elements
        assert torch.isnan(seg_loss).any() == False, f"Found NaNs in seg_loss: {seg_loss}"
        total_loss += seg_loss
        loss_dict["loss_segmentation"] = seg_loss

        loss_dict["loss_objective"] = total_loss

        return loss_dict