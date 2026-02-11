# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np
import random


from data.dataset_util import *
from data.base_dataset import BaseDataset

def convert_joint_type(joint_type: str) -> int:
    joint_type = joint_type.strip().lower()
    if joint_type == "revolute" or joint_type == "continuous":
        return 0
    elif joint_type == "prismatic":
        return 1
    else:
        assert False, f"Unsupported joint type: {joint_type}"


class PMDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        PM_DIR: str = None,
        PM_ANNOTATION_PATH: str = None,
    ):
        """
        Initialize the PMDataset.

        Args:
            common_conf: Configuration object with common settings.
        Raises:
            ValueError: If PM_DIR or PM_ANNOTATION_PATH is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        annotations = {}
        self.PM_DIR: str = PM_DIR
        self.PM_ANNOTATION_PATH: str = PM_ANNOTATION_PATH
        with open(PM_ANNOTATION_PATH, "r") as f:
            annotations: dict[str, list] = json.load(f)
        self.split_paths: list[str] = annotations[split]
        self.len_train = len(self.split_paths)

    def get_data(
        self,
        seq_index: int,
        img_per_seq: int = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        cur_path: str = self.split_paths[seq_index]
        sample_id: str = os.path.basename(cur_path)
        json_path: str = os.path.join(cur_path, "mobility_flat.json")
        with open(json_path, "r") as f:
            sample_json = json.load(f)
        num_images = len(sample_json) + 1 # all joint-state images + rest-state image
        if img_per_seq is not None and img_per_seq < num_images:
            assert img_per_seq > 1, "cannot < 2, explanation right below"
            num_images = img_per_seq
        assert num_images > 1, "each sample requires >= 2 images, 1 at rest state and 1 movable"
        target_image_shape = self.get_target_shape(aspect_ratio)
        
        images = []
        seg_images = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []
        joint_names = []
        joint_types = []
        joint_axes = []
        joint_origins = []
        revolute_origin_masks = [] # [2026-02-11] @tcm: store 1s for origins of revolute joint, 0s for prismatic's
        joint_ranges = []
        sample_json_pairs = list(sample_json.items())
        sample_indices = random.sample(range(len(sample_json_pairs)), num_images - 1) # [2026-02-11] @tcm: for movables
        sample_indices = [-1] + sample_indices

        for idx in sample_indices:
            
            if idx == -1:
                joint_name = "rest"
                joint_info = sample_json_pairs[sample_indices[1]][1] # [2026-02-11] @tcm: just take from next guy as place-holder
                image_path = os.path.join(self.PM_DIR, sample_id, "render", "rest.png")
                seg_path = os.path.join(self.PM_DIR, sample_id, "render", "rest_seg.png")
            else:
                joint_name, joint_info = sample_json_pairs[idx]
                image_path = os.path.join(self.PM_DIR, sample_id, "render", f"{joint_name}.png")
                seg_path = os.path.join(self.PM_DIR, sample_id, "render", f"{joint_name}_seg.png")
            
            image = read_image_cv2(image_path)            
            seg_image = read_image_cv2(seg_path)
            original_size = np.array(image.shape[:2])
            joint_type = convert_joint_type(joint_info["type"])
            revolute_origin = 1 if joint_type == 0 else 0
            joint_axis = joint_info["axis"]
            joint_origin = joint_info["origin"]
            joint_motion_lower = joint_info["motion_lower"]
            joint_motion_upper = joint_info["motion_upper"]
            intri_opencv = np.array(joint_info["camera_intrinsics"])
            extri_opencv = np.array(joint_info["camera_extrinsics"])
            (
                image,
                _, # depth_map
                seg_image,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                None,
                seg_image,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=None,
            )
            images.append(image)
            seg_images.append(seg_image)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            image_paths.append(image_path)
            original_sizes.append(original_size)
            joint_names.append(joint_name)
            revolute_origin_masks.append(revolute_origin)
            joint_types.append(np.array(joint_type))
            joint_axes.append(np.array(joint_axis))
            joint_origins.append(np.array(joint_origin))
            joint_ranges.append((joint_motion_lower, joint_motion_upper))

        batch = {
            "sample_id": sample_id,
            "images": images,
            "seg_images": seg_images,
            "image_paths": image_paths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "original_sizes": original_sizes,
            "joint_names": joint_names,
            "revolute_origin_masks": revolute_origin_masks,
            "joint_types": joint_types,
            "joint_axes": joint_axes,
            "joint_origins": joint_origins,
            "joint_ranges": joint_ranges
        }
        return batch

            
