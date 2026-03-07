# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment configs for training Cosmos video model from LeRobot datasets.

Usage:
  torchrun --nproc_per_node=8 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment=predict2_video2world_training_2b_cosmos_lerobot_libero_flat_short
"""

import math

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
# from cosmos_predict2._src.predict2.callbacks.autoregressive_video_gen import AutoregressiveVideoGen
from cosmos_predict2._src.predict2.callbacks.save_demo_data import SaveDemoData
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video_lerobot import (
    LeRobotVideoDatasetFlat,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video_droid import (
    VideoDatasetFlat,
    get_generic_weighted_dataloader,
    get_weighted_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey, ModelSize

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]
DEFAULT_CHECKPOINT_14B = MODEL_CHECKPOINTS[ModelKey(post_trained=False, size=ModelSize._14B)]


# ---------------------------------------------------------------------------
# Training config automation: LR scaling + max_iter from dataset size
# ---------------------------------------------------------------------------
# Dataset metadata: {name: (num_episodes, avg_episode_length, frame_skip)}
DATASET_INFO = {
    "libero":  (1693,  162, 1),
    "droid":   (57774, 255, 2),
    "robocasa": (1199, 264, 2),
    "yam":     (92,    448, 4),
}
NUM_FRAMES_PER_SAMPLE = 17
REFERENCE_EFFECTIVE_BATCH = 8  # LR was tuned for this effective batch size


def compute_training_params(
    dataset_name: str,
    batch_size_per_gpu: int,
    num_gpus: int = 8,
    base_lr: float = 2 ** (-14.5),
    target_full_passes: int = 15,
    min_iter: int = 5000,
    max_iter_cap: int = 80000,
) -> dict:
    """Compute LR (linear scaling) and max_iter (target full-coverage passes).

    A "full-coverage pass" = enough samples to statistically cover every frame
    in every episode once (accounting for frame_skip and num_frames_per_sample).

    Args:
        base_lr: Reference LR tuned for REFERENCE_EFFECTIVE_BATCH (8).
        target_full_passes: How many times to cover every frame in every episode.
        min_iter: Floor on max_iter (diffusion models need enough steps to converge).
        max_iter_cap: Ceiling on max_iter to keep training time bounded.
    """
    num_episodes, avg_ep_len, frame_skip = DATASET_INFO[dataset_name]
    effective_batch = batch_size_per_gpu * num_gpus

    # Linear LR scaling
    lr = base_lr * (effective_batch / REFERENCE_EFFECTIVE_BATCH)

    # Samples needed per episode to cover all its (subsampled) frames
    effective_ep_len = avg_ep_len / frame_skip
    samples_per_episode = max(1, effective_ep_len / NUM_FRAMES_PER_SAMPLE)

    # Total steps for target full-coverage passes, clamped to [min_iter, max_iter_cap]
    raw_iter = target_full_passes * num_episodes * samples_per_episode / effective_batch
    max_iter = int(min(max_iter_cap, max(min_iter, raw_iter)))
    # Round to nearest 1000
    max_iter = max(1000, round(max_iter / 1000) * 1000)

    # Save ~10 checkpoints per run (min 500)
    save_iter = max(500, round(max_iter / 10 / 500) * 500)

    # Warmup = ~5% of training, capped at [500, 3000]
    warm_up_steps = min(3000, max(500, max_iter // 20))

    return dict(lr=lr, max_iter=max_iter, save_iter=save_iter, warm_up_steps=warm_up_steps)


# ---------------------------------------------------------------------------
# LIBERO dataset (2 cameras: image + wrist_image)
# ---------------------------------------------------------------------------
example_lerobot_libero_short = L(LeRobotVideoDatasetFlat)(
    repo_id="physical-intelligence/libero",
    num_frames=17,
    video_size=(224, 224 * 2),
    camera_keys=None,  # Auto-detect from dataset
    layout="horizontal",  # 2 cameras → horizontal concat
)

dataloader_train_lerobot_libero_short = L(get_generic_dataloader)(
    dataset=example_lerobot_libero_short,
    sampler=L(get_sampler)(dataset=example_lerobot_libero_short),
    batch_size=16,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# ---------------------------------------------------------------------------
# DROID dataset (3 cameras: exterior_image_1_left, exterior_image_2_left, wrist_image_left)
# ---------------------------------------------------------------------------
example_lerobot_droid_short = L(LeRobotVideoDatasetFlat)(
    repo_id="GEAR-Dreams/DreamZero-DROID-Data",
    num_frames=17,
    video_size=(180, 320 * 3),  # Native 320x180 per camera (16:9), 3 cameras → 960x180
    camera_keys=None,  # Auto-detect
    layout="horizontal",  # 3 cameras → horizontal flatten
    frame_skip=2,
)

# Polaris DROID dataset (local video format, same resolution as lerobot DROID)
example_polaris_droid_short = L(VideoDatasetFlat)(
    dataset_dir="/k8s-nfs/personal/mishutk/polaris_droid_video",
    num_frames=17,
    video_size=(180, 320 * 3),
    frame_skip=2,
)

dataloader_train_lerobot_droid_short = L(get_generic_weighted_dataloader)(
    datasets=[
        example_lerobot_droid_short,
        example_polaris_droid_short,
    ],
    weights=[0.95, 0.05],
    sampler=L(get_weighted_sampler)(
        datasets=[
            example_lerobot_droid_short,
            example_polaris_droid_short,
        ],
        weights=[0.95, 0.05],
    ),
    batch_size=8,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# ---------------------------------------------------------------------------
# ROBOCASA dataset (3 cameras: robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand)
# ---------------------------------------------------------------------------
example_lerobot_robocasa_short = L(LeRobotVideoDatasetFlat)(
    repo_id="imishutk0410/robocasa_noops",
    num_frames=17,
    video_size=(224, 224 * 3),
    camera_keys=None,  # Auto-detect
    layout="horizontal",  # 3 cameras → horizontal flatten
    frame_skip=2,
)

dataloader_train_lerobot_robocasa_short = L(get_generic_dataloader)(
    dataset=example_lerobot_robocasa_short,
    sampler=L(get_sampler)(dataset=example_lerobot_robocasa_short),
    batch_size=8,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)


# ---------------------------------------------------------------------------
# YAM dataset (3 cameras: left_image, left_wrist_left, right_wrist_left)
# ---------------------------------------------------------------------------
example_lerobot_yam_short = L(LeRobotVideoDatasetFlat)(
    repo_id="imishutk0410/yam-pnp-apple-toy",
    num_frames=17,
    video_size=(240, 320 * 3),  # Half native 640x480 per camera (4:3), 3 cameras → 960x240
    camera_keys=None,  # Auto-detect
    layout="horizontal",  # 3 cameras → horizontal flatten
    frame_skip=4,
    video_backend="pyav",  # av1 codec not supported by torchcodec
)

dataloader_train_lerobot_yam_short = L(get_generic_dataloader)(
    dataset=example_lerobot_yam_short,
    sampler=L(get_sampler)(dataset=example_lerobot_yam_short),
    batch_size=4,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)


# ---------------------------------------------------------------------------
# Experiment: LIBERO
# ---------------------------------------------------------------------------
_libero_2b = compute_training_params("libero", batch_size_per_gpu=16, base_lr=2 ** (-14.5), target_full_passes=100)
predict2_video2world_training_2b_cosmos_lerobot_libero_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_lerobot_libero_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_libero_short,
    checkpoint=dict(
        save_iter=_libero_2b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_libero_2b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_libero_2b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_libero_2b["max_iter"],
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)

# ---------------------------------------------------------------------------
# Experiment: DROID
# ---------------------------------------------------------------------------
_droid_2b = compute_training_params("droid", batch_size_per_gpu=8, base_lr=2 ** (-14.5))
predict2_video2world_training_2b_cosmos_lerobot_droid_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_lerobot_droid_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_droid_short,
    checkpoint=dict(
        save_iter=_droid_2b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_droid_2b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_droid_2b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_droid_2b["max_iter"],
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# ---------------------------------------------------------------------------
# Experiment: ROBOCASA
# ---------------------------------------------------------------------------
_robocasa_2b = compute_training_params("robocasa", batch_size_per_gpu=8, base_lr=2 ** (-14.5), target_full_passes=100)
predict2_video2world_training_2b_cosmos_lerobot_robocasa_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_lerobot_robocasa_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_robocasa_short,
    checkpoint=dict(
        save_iter=_robocasa_2b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_robocasa_2b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_robocasa_2b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_robocasa_2b["max_iter"],
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# ---------------------------------------------------------------------------
# YAM dataset with wrist blackout (3 cameras, wrist views periodically blacked out)
# ---------------------------------------------------------------------------
example_lerobot_yam_wrist_blackout_short = L(LeRobotVideoDatasetFlat)(
    repo_id="imishutk0410/yam-pnp-apple-toy",
    num_frames=17,
    video_size=(240, 320 * 3),  # Half native 640x480 per camera (4:3), 3 cameras → 960x240
    camera_keys=None,  # Auto-detect
    layout="horizontal",  # 3 cameras → horizontal flatten
    frame_skip=4,
    video_backend="pyav",  # av1 codec not supported by torchcodec
    wrist_blackout_prob=0.5,  # Black out wrist views 50% of the time
)

dataloader_train_lerobot_yam_wrist_blackout_short = L(get_generic_dataloader)(
    dataset=example_lerobot_yam_wrist_blackout_short,
    sampler=L(get_sampler)(dataset=example_lerobot_yam_wrist_blackout_short),
    batch_size=4,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)


# ---------------------------------------------------------------------------
# Experiment: YAM
# ---------------------------------------------------------------------------
_yam_2b = compute_training_params("yam", batch_size_per_gpu=4, base_lr=2 ** (-14.5))
predict2_video2world_training_2b_cosmos_lerobot_yam_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_lerobot_yam_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_yam_short,
    checkpoint=dict(
        save_iter=_yam_2b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_yam_2b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_yam_2b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_yam_2b["max_iter"],
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# ---------------------------------------------------------------------------
# Experiment: YAM with wrist blackout
# ---------------------------------------------------------------------------
predict2_video2world_training_2b_cosmos_lerobot_yam_wrist_blackout_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_lerobot_yam_wrist_blackout_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_yam_wrist_blackout_short,
    checkpoint=dict(
        save_iter=_yam_2b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_yam_2b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_yam_2b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_yam_2b["max_iter"],
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# ===========================================================================
# 14B EXPERIMENTS
# ===========================================================================
_BASE_LR_14B = 2 ** (-15.5)  # Lower base LR for 14B (larger model → more sensitive)

# ---------------------------------------------------------------------------
# Experiment: LIBERO 14B
# ---------------------------------------------------------------------------
_libero_14b = compute_training_params("libero", batch_size_per_gpu=16, base_lr=_BASE_LR_14B, target_full_passes=100)
predict2_video2world_training_14b_cosmos_lerobot_libero_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_14B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="14b_cosmos_lerobot_libero_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_libero_short,
    checkpoint=dict(
        save_iter=_libero_14b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_14B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_libero_14b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_libero_14b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_libero_14b["max_iter"],
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)

# ---------------------------------------------------------------------------
# Experiment: YAM 14B
# ---------------------------------------------------------------------------
_yam_14b = compute_training_params("yam", batch_size_per_gpu=4, base_lr=_BASE_LR_14B)
predict2_video2world_training_14b_cosmos_lerobot_yam_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_14B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="14b_cosmos_lerobot_yam_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_yam_short,
    checkpoint=dict(
        save_iter=_yam_14b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_14B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_yam_14b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_yam_14b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_yam_14b["max_iter"],
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)

# ---------------------------------------------------------------------------
# Experiment: YAM 14B with wrist blackout
# ---------------------------------------------------------------------------
predict2_video2world_training_14b_cosmos_lerobot_yam_wrist_blackout_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_14B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="14b_cosmos_lerobot_yam_wrist_blackout_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_yam_wrist_blackout_short,
    checkpoint=dict(
        save_iter=_yam_14b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_14B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_yam_14b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_yam_14b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_yam_14b["max_iter"],
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# ---------------------------------------------------------------------------
# Experiment: DROID 14B
# ---------------------------------------------------------------------------
_droid_14b = compute_training_params("droid", batch_size_per_gpu=8, base_lr=_BASE_LR_14B)
predict2_video2world_training_14b_cosmos_lerobot_droid_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_14B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="14b_cosmos_lerobot_droid_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_droid_short,
    checkpoint=dict(
        save_iter=_droid_14b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_14B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_droid_14b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_droid_14b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_droid_14b["max_iter"],
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)

# ---------------------------------------------------------------------------
# Experiment: ROBOCASA 14B
# ---------------------------------------------------------------------------
_robocasa_14b = compute_training_params("robocasa", batch_size_per_gpu=8, base_lr=_BASE_LR_14B, target_full_passes=100)
predict2_video2world_training_14b_cosmos_lerobot_robocasa_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_14B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="14b_cosmos_lerobot_robocasa_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_robocasa_short,
    checkpoint=dict(
        save_iter=_robocasa_14b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_14B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_robocasa_14b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_robocasa_14b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_robocasa_14b["max_iter"],
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


cs = ConfigStore.instance()

for _item in [
    predict2_video2world_training_2b_cosmos_lerobot_libero_flat_short,
    predict2_video2world_training_2b_cosmos_lerobot_droid_flat_short,
    predict2_video2world_training_2b_cosmos_lerobot_robocasa_flat_short,
    predict2_video2world_training_2b_cosmos_lerobot_yam_flat_short,
    predict2_video2world_training_2b_cosmos_lerobot_yam_wrist_blackout_flat_short,
    predict2_video2world_training_14b_cosmos_lerobot_libero_flat_short,
    predict2_video2world_training_14b_cosmos_lerobot_droid_flat_short,
    predict2_video2world_training_14b_cosmos_lerobot_robocasa_flat_short,
    predict2_video2world_training_14b_cosmos_lerobot_yam_flat_short,
    predict2_video2world_training_14b_cosmos_lerobot_yam_wrist_blackout_flat_short,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
