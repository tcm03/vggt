
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator_articulate import AggregatorArticulate
from vggt.heads.axis_head import AxisHead
from vggt.heads.cls_head import ClsHead
from vggt.heads.dpt_head import DPTHead


class VGGTiculate(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        img_size=518, 
        patch_size=14, 
        embed_dim=1024,
        enable_axis=True, 
        enable_segmentation=True, 
    ):
        super().__init__()

        self.aggregator = AggregatorArticulate(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.cls_head = ClsHead(dim_in = 2 * embed_dim)

        self.axis_head = AxisHead(dim_in=2 * embed_dim) if enable_axis else None
        # self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.segmentation_head = DPTHead(
            dim_in=2 * embed_dim, 
            output_dim=3, # [2026-02-06] @tcm: 3 labels (background, base, artic. part), no confidence (but this seems to not be used when feature_only=True), but I make it useful even when feature_only=True (see DPTHead impl)
            # activation="log_softmax", # [2026-02-11] @tcm: wouldn't be used if feature_only=True
            feature_only=True # [2026-02-06] @tcm: don't need confidence for segmentation
        ) if enable_segmentation else None

    def forward(self, images: torch.Tensor):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Axis pose encoding with shape [B, S, 9] (from the last iteration)
                - segmentation (torch.Tensor): Predicted segmentation maps with shape [B, S, H, W, 1]
                - segmentation_conf (torch.Tensor): Confidence scores for segmentation predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images) # [2026-01-26] @tcm: aggregated_tokens_list = [torch.Size[1, B=25, S=930, D=2048]] x L=24

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):

            if self.cls_head is not None:
                cls_logits = self.cls_head(aggregated_tokens_list)
                predictions["cls_logits"] = cls_logits

            if self.axis_head is not None:
                pose_enc_list = self.axis_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.segmentation_head is not None:
                segmentation = self.segmentation_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["segmentation"] = segmentation
                # predictions["segmentation_conf"] = segmentation_conf

            # if self.point_head is not None:
            #     pts3d, pts3d_conf = self.point_head(
            #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            #     )
            #     predictions["world_points"] = pts3d
            #     predictions["world_points_conf"] = pts3d_conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

