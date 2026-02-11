# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose_axis


class ClsHead(nn.Module):
    """
    ClsHead predicts articulation type token representations

    It applies a lightweight MLP to dedicated axis tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048
    ):
        super().__init__()

        self.target_dim = 2 # for now, assume 2 joint types: prismatic, and revolute

        # Normalizations for cls token
        self.token_norm = nn.LayerNorm(dim_in)

        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list) -> list:
        """
        Forward pass to predict axis parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.

        Returns:
            cls_logits
        """
        # Use tokens from the last block for axis prediction.
        tokens = aggregated_tokens_list[-1] # [2026-01-26] @tcm: tokens.shape = [B, S, N, D=2048]
        assert tokens.ndim == 4 and tokens.shape[1] > 1, f"expected shape: (B, S, N, D), found {tokens.shape}; expect S dim > 1 view"
        tokens = tokens[:, 1:, ...] # [2026-02-11] @tcm: don't predict joint type for 1st view (rest state)

        # Extract the cls tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens) # [2026-01-26] @tcm: pose_tokens.shape = [B, S-1, D=2048]

        cls_logits = self.pose_branch(pose_tokens)
        return cls_logits