# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional
import torch

from hydra.utils import instantiate
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler
from abc import ABC, abstractmethod

from .worker_fn import get_worker_init_fn

IGNORE_INDEX: int = -100


def pm_collate_fn(batch: list[dict]):
    # [2026-02-10] @tcm: batch: 5 x [dict(
    # sample_id: int, 
    # images: torch.Size([Si, 3, 476, 518]), 
    # seg_images: torch.Size([Si, 476, 518]), 
    # extrinsics: torch.Size([Si, 3, 4]), 
    # intrinsics: torch.Size([Si, 3, 3]), 
    # joint_names: list[str], 
    # joint_types: list[str], 
    # joint_axes: torch.Size([Si, 3]), 
    # joint_origins: torch.Size([Si, 3]), 
    # joint_ranges: torch.Size([Si, 2])
    # )]
    
    max_s = max(item["images"].shape[0] for item in batch)
    collated_batch = {
        "sample_ids": [],
        "images": [],
        "seg_images": [],
        "extrinsics": [],
        "intrinsics": [],
        "joint_names": [],
        "joint_types": [],
        "joint_axes": [],
        "joint_origins": [],
        "joint_ranges": [],
        "nonpad_masks": [], # [2026-02-11] @tcm: shape (B, S), True entries indicate padded elements
        "revolute_origin_masks": []
    }

    for item in batch:
        s_i = item["images"].shape[0]
        pad_amount = max_s - s_i
        collated_batch["sample_ids"].append(item["sample_id"])

        nonpad_mask = torch.cat([
            torch.ones(s_i, dtype=torch.long), 
            torch.zeros(pad_amount, dtype=torch.long)
        ])
        collated_batch["nonpad_masks"].append(nonpad_mask)
        # Images: [Si, 3, H, W] -> Pad dim 0
        if pad_amount > 0:
            img_pad = torch.zeros((pad_amount, *item["images"].shape[1:]), dtype=item["images"].dtype)
            padded_images = torch.cat([item["images"], img_pad], dim=0)

            seg_img_pad = torch.full((pad_amount, *item["seg_images"].shape[1:]), IGNORE_INDEX, dtype=item["seg_images"].dtype)
            padded_seg_images = torch.cat([item["seg_images"], seg_img_pad], dim=0)
            
            ext_pad = torch.zeros((pad_amount, *item["extrinsics"].shape[1:]), dtype=item["extrinsics"].dtype)
            padded_extrinsics = torch.cat([item["extrinsics"], ext_pad], dim=0)
            
            int_pad = torch.zeros((pad_amount, *item["intrinsics"].shape[1:]), dtype=item["intrinsics"].dtype)
            padded_intrinsics = torch.cat([item["intrinsics"], int_pad], dim=0)

            ax_pad = torch.zeros((pad_amount, *item["joint_axes"].shape[1:]), dtype=item["joint_axes"].dtype)
            padded_axes = torch.cat([item["joint_axes"], ax_pad], dim=0)

            org_pad = torch.zeros((pad_amount, *item["joint_origins"].shape[1:]), dtype=item["joint_origins"].dtype)
            padded_origins = torch.cat([item["joint_origins"], org_pad], dim=0)

            rng_pad = torch.zeros((pad_amount, *item["joint_ranges"].shape[1:]), dtype=item["joint_ranges"].dtype)
            # [2026-02-11] @tcm: is padding safe here to compute loss?
            padded_ranges = torch.cat([item["joint_ranges"], rng_pad], dim=0)
            
            padded_names = item["joint_names"] + [""] * pad_amount
            
            padded_types = torch.cat([item["joint_types"], torch.tensor([IGNORE_INDEX] * pad_amount)])

            padded_revolute_origin_masks = torch.cat([item["revolute_origin_masks"], torch.zeros(pad_amount, dtype=item["revolute_origin_masks"].dtype)])

        else:
            padded_images = item["images"]
            padded_seg_images = item["seg_images"]
            padded_extrinsics = item["extrinsics"]
            padded_intrinsics = item["intrinsics"]
            padded_axes = item["joint_axes"]
            padded_origins = item["joint_origins"]
            padded_ranges = item["joint_ranges"]
            padded_names = item["joint_names"]
            padded_types = item["joint_types"]
            padded_revolute_origin_masks = item["revolute_origin_masks"]

        collated_batch["images"].append(padded_images)
        collated_batch["seg_images"].append(padded_seg_images)
        collated_batch["extrinsics"].append(padded_extrinsics)
        collated_batch["intrinsics"].append(padded_intrinsics)
        collated_batch["joint_axes"].append(padded_axes)
        collated_batch["joint_origins"].append(padded_origins)
        collated_batch["joint_ranges"].append(padded_ranges)
        collated_batch["joint_names"].append(padded_names)
        collated_batch["joint_types"].append(padded_types)
        collated_batch["revolute_origin_masks"].append(padded_revolute_origin_masks)

    collated_batch["images"] = torch.stack(collated_batch["images"], dim=0)
    collated_batch["seg_images"] = torch.stack(collated_batch["seg_images"], dim=0)
    collated_batch["extrinsics"] = torch.stack(collated_batch["extrinsics"], dim=0)
    collated_batch["intrinsics"] = torch.stack(collated_batch["intrinsics"], dim=0)
    collated_batch["joint_axes"] = torch.stack(collated_batch["joint_axes"], dim=0)
    collated_batch["joint_origins"] = torch.stack(collated_batch["joint_origins"], dim=0)
    collated_batch["joint_ranges"] = torch.stack(collated_batch["joint_ranges"], dim=0)
    collated_batch["nonpad_masks"] = torch.stack(collated_batch["nonpad_masks"], dim=0)
    collated_batch["joint_types"] = torch.stack(collated_batch["joint_types"], dim=0)
    collated_batch["revolute_origin_masks"] = torch.stack(collated_batch["revolute_origin_masks"], dim=0)
    return collated_batch


class DynamicTorchDataset(ABC):
    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
        max_img_per_gpu: int = 48,
    ) -> None:
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = pm_collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu

        # Instantiate the dataset
        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False) # [2026-02-10] @tcm: self.dataset: data.articulate_composed_dataset.ArticulateComposedDataset

        # Extract aspect ratio and image number ranges from the configuration
        self.aspect_ratio_range = common_config.augs.aspects  # e.g., [0.5, 1.0]
        self.image_num_range = common_config.img_nums    # e.g., [2, 24]

        # Validate the aspect ratio and image number ranges
        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        # Create samplers
        self.sampler = DynamicDistributedSampler(self.dataset, seed=seed, shuffle=shuffle)
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=seed,
            max_img_per_gpu=max_img_per_gpu,
            drop_last=True
        )

    def get_loader(self, epoch):
        print("Building dynamic dataloader with epoch:", epoch)

        # Set the epoch for the sampler
        self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )
        

class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 aspect_ratio_range,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48,
                 drop_last=False # [2026-02-11] @tcm: drop last batch
    ):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.rng = random.Random()

        self.drop_last = drop_last

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number and aspect ratio
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

                # Update sampler parameters
                self.sampler.update_parameters(
                    aspect_ratio=random_aspect_ratio,
                    image_num=random_image_num
                )

                # Calculate batch size based on max images per GPU and current image number
                batch_size = self.max_img_per_gpu / random_image_num
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, image_num, aspect_ratio)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples
                # [2026-02-10] @tcm: e.g., (37542, 9, 0.92), (60279, 9, 0.92), (68544, 9, 0.92), (70051, 9, 0.92), (15790, 9, 0.92) => 5 scenes, 9 frames/ scene => 45 total frames
                if not current_batch:
                    break  # No more data to yield
                
                if len(current_batch) < batch_size:
                    if self.drop_last:
                        break

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last # [2026-02-11] @tcm: drop indices from dataset to ensure total # samples divisible by # gpus in distributed training
        )
        self.aspect_ratio = None
        self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num, self.aspect_ratio,)

    def update_parameters(self, aspect_ratio, image_num):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num
