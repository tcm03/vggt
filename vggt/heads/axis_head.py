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


class AxisHead(nn.Module):
    """
    AxisHead predicts axis' position and orientation from token representations

    It applies a series of transformer blocks (the "trunk") to dedicated axis tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048
    ):
        super().__init__()

        self.target_dim = 6

        # Normalizations for axis token
        self.token_norm = nn.LayerNorm(dim_in)

        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict axis parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted axis encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for axis prediction.
        tokens = aggregated_tokens_list[-1] # [2026-01-26] @tcm: tokens.shape = [1, B=25, S=930, D=2048]

        # Extract the axis tokens
        pose_tokens = tokens[:, :, 1]
        pose_tokens = self.token_norm(pose_tokens) # [2026-01-26] @tcm: pose_tokens.shape = [1, B=25, D=2048]
        pred_pose_enc_list = [self.pose_branch(pose_tokens)]
        return pred_pose_enc_list


# [2026-02-11] @tcm: This is AxisHead that is slightly modified from CameraHead, hide it for now because of time
# class AxisHead(nn.Module):
#     """
#     AxisHead predicts axis' position and orientation from token representations

#     It applies a series of transformer blocks (the "trunk") to dedicated axis tokens.
#     """

#     def __init__(
#         self,
#         dim_in: int = 2048,
#         trunk_depth: int = 4,
#         pose_encoding_type: str = "absT_absR",
#         num_heads: int = 16,
#         mlp_ratio: int = 4,
#         init_values: float = 0.01,
#         trans_act: str = "linear",
#         quat_act: str = "linear",
#     ):
#         super().__init__()

#         if pose_encoding_type == "absT_absR":
#             self.target_dim = 6
#         else:
#             raise ValueError(f"Unsupported axis encoding type: {pose_encoding_type}")

#         self.trans_act = trans_act
#         self.quat_act = quat_act
#         self.trunk_depth = trunk_depth # [2026-01-26] @tcm: self.trunk_depth = 4

#         # Build the trunk using a sequence of transformer blocks.
#         self.trunk = nn.Sequential(
#             *[
#                 Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
#                 for _ in range(trunk_depth)
#             ]
#         )

#         # Normalizations for camera token and trunk output.
#         self.token_norm = nn.LayerNorm(dim_in)
#         self.trunk_norm = nn.LayerNorm(dim_in)

#         # Learnable empty camera pose token.
#         self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim)) # [2026-01-26] @tcm: self.empty_pose_tokens.shape = [1, 1, 6]
#         self.embed_pose = nn.Linear(self.target_dim, dim_in) # [2026-01-26] @tcm: Linear(in_features=6, out_features=2048, bias=True)

#         # Module for producing modulation parameters: shift, scale, and a gate.
#         self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

#         # Adaptive layer normalization without affine parameters.
#         self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
#         self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

#     def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
#         """
#         Forward pass to predict axis parameters.

#         Args:
#             aggregated_tokens_list (list): List of token tensors from the network;
#                 the last tensor is used for prediction.
#             num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

#         Returns:
#             list: A list of predicted axis encodings (post-activation) from each iteration.
#         """
#         # Use tokens from the last block for axis prediction.
#         tokens = aggregated_tokens_list[-1] # [2026-01-26] @tcm: tokens.shape = [1, B=25, S=930, D=2048]

#         # Extract the axis tokens
#         pose_tokens = tokens[:, :, 1]
#         pose_tokens = self.token_norm(pose_tokens) # [2026-01-26] @tcm: pose_tokens.shape = [1, B=25, D=2048]

#         pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
#         return pred_pose_enc_list

#     def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
#         """
#         Iteratively refine axis pose predictions.

#         Args:
#             pose_tokens (torch.Tensor): Normalized axis tokens with shape [B, S, C].
#             num_iterations (int): Number of refinement iterations.

#         Returns:
#             list: List of activated axis encodings from each iteration.
#         """
#         B, S, C = pose_tokens.shape
#         pred_pose_enc = None
#         pred_pose_enc_list = []

#         for _ in range(num_iterations): # [2026-01-26] @tcm: num_iterations=4
#             # Use a learned empty pose for the first iteration.
#             if pred_pose_enc is None:
#                 module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1)) # [2026-01-26] @tcm: module_input.shape = [1, B=25, D=2048]
#             else:
#                 # Detach the previous prediction to avoid backprop through time.
#                 pred_pose_enc = pred_pose_enc.detach()
#                 module_input = self.embed_pose(pred_pose_enc)

#             # Generate modulation parameters and split them into shift, scale, and gate components.
#             shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1) # [2026-01-26] @tcm: all's shape: [1, B=25, D=2048]

#             # Adaptive layer normalization and modulation.
#             pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa) # [2026-01-26] @tcm: pose_tokens_modulated.shape = [1, B=25, D=2048]
#             pose_tokens_modulated = pose_tokens_modulated + pose_tokens

#             pose_tokens_modulated = self.trunk(pose_tokens_modulated) # [2026-01-26] @tcm: pose_tokens_modulated.shape = [1, B=25, D=2048]
#             # Compute the delta update for the pose encoding.
#             pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated)) # [2026-01-26] @tcm: pred_pose_enc_delta.shape = [1, B=25, D=6]

#             if pred_pose_enc is None:
#                 pred_pose_enc = pred_pose_enc_delta # [2026-01-26] @tcm: pred_pose_enc.shape = [1, B=25, D=6]
#             else:
#                 pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

#             # Apply final activation functions for translation, quaternion.
#             activated_pose = activate_pose_axis(
#                 pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act
#             ) # [2026-01-26] @tcm: activated_pose.shape = [B=1, S=25, 6]
#             pred_pose_enc_list.append(activated_pose)

#         return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift