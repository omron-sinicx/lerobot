#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process Force Diffusion data stored in Zarr format"""

import logging
import shutil
from pathlib import Path

import torch
import tqdm
import zarr
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub._umi_imagecodecs_numcodecs import register_codecs
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def check_format(raw_dir) -> bool:
    zarr_path = raw_dir / "wipingdiffusion.zarr" #change this later 
    zarr_data = zarr.open(zarr_path, mode="r")

    image_keys = ['data/' + k for k in list(zarr_data['data']) if k.startswith("img")]

    required_datasets = {
        "data/action",
        # "data/img", we are going to add image_keys since we have multiple cameras
        "data/state",
        "data/ft",
        "meta/episode_ends",
    }
    required_datasets.update(image_keys)

    for dataset in required_datasets:
        if dataset not in zarr_data:
            return False

    # mandatory to access zarr_data
    register_codecs()
    nb_frames = zarr_data[image_keys[0]].shape[0]

    required_datasets.remove("meta/episode_ends")
    assert all(nb_frames == zarr_data[dataset].shape[0] for dataset in required_datasets)


def load_from_raw(raw_dir: Path, videos_dir: Path, fps: int, video: bool, episodes: list[int] | None = None):
    zarr_path = raw_dir / "wipingdiffusion.zarr"
    zarr_data = zarr.open(zarr_path, mode="r")
    img_keys = [] # to be returned with the fucntion in the final representation 
    # We process the image data separately because it is too large to fit in memory
    actions = torch.from_numpy(zarr_data["data/action"][:])
    states = torch.from_numpy(zarr_data["data/state"][:])
    fts = torch.from_numpy(zarr_data["data/ft"][:])

    episode_ends = zarr_data["meta/episode_ends"][:]
    num_episodes = episode_ends.shape[0]

    # We convert it in torch tensor later because the jit function does not support torch tensors
    # episode_ends = torch.from_numpy(episode_ends)
    episode_ends = torch.tensor(episode_ends, dtype=torch.long)

    # load data indices from which each episode starts and ends
    from_ids, to_ids = [], []
    from_idx = 0
    for to_idx in episode_ends:
        from_ids.append(from_idx)
        to_ids.append(to_idx)
        from_idx = to_idx

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    
    for ep_idx, selected_ep_idx in tqdm.tqdm(enumerate(ep_ids)):
        from_idx = from_ids[selected_ep_idx]
        to_idx = to_ids[selected_ep_idx]
        num_frames = to_idx - from_idx

        # TODO(rcadene): save temporary images of the episode?

        # print(f" from idx : {from_idx}")
        print(f" the type {type(to_idx)} and the value {to_idx} ")

        state = states[from_idx:to_idx]

        ep_dict = {}
        #get the keys for the cameras in the dataset
        image_keys = ['data/' + k for k in list(zarr_data['data']) if k.startswith("img")]
        num_cameras = len(image_keys)
        
        for image_key in image_keys:
            # load 57MB of images in RAM (400x224x224x3 uint8)
            imgs_array = zarr_data[image_key][from_idx:to_idx]
            camera_name = image_key.split("/")[1]
            img_key = "observation.image." + camera_name
            img_keys.append(img_key)
            if video:
                # save png images in temporary directory
                tmp_imgs_dir = videos_dir / "tmp_images"
                save_images_concurrently(imgs_array, tmp_imgs_dir)

                # encode images to a mp4 video
                fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                video_path = videos_dir / fname
                encode_video_frames(tmp_imgs_dir, video_path, fps)

                # clean temporary images directory
                shutil.rmtree(tmp_imgs_dir)

                # store the reference to the video frame
                ep_dict[img_key] = [{"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)]
            else:
                ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

        ep_dict["observation.state"] = state
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["episode_data_index_from"] = torch.tensor([from_idx] * num_frames)
        ep_dict["episode_data_index_to"] = torch.tensor([from_idx + num_frames] * num_frames)
        ep_dict["action"] = actions[from_idx:to_idx]
        ep_dict["observation.ft"] = fts[from_idx:to_idx]
        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict, img_keys


def to_hf_dataset(data_dict, video, img_keys):
    features = {}

    for img_key in img_keys:
        if video:
            features[img_key] = VideoFrame()
        else:
            features[img_key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["index"] = Value(dtype="int64", id=None)
    features["episode_data_index_from"] = Value(dtype="int64", id=None)
    features["episode_data_index_to"] = Value(dtype="int64", id=None)
    # `start_pos` and `end_pos` respectively represent the positions of the end-effector
    # at the beginning and the end of the episode.
    # `gripper_width` indicates the distance between the grippers, and this value is included
    # in the state vector, which comprises the concatenation of the end-effector position
    # and gripper width.
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["observation.ft"] = Sequence(
        length=data_dict["observation.ft"].shape[1], feature=Value(dtype="float32", id=None)
    )

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        # For umi cup in the wild: https://arxiv.org/pdf/2402.10329#table.caption.16
        fps = 10

    if not video:
        logging.warning(
            "Generating UMI dataset without `video=True` creates ~150GB on disk and requires ~80GB in RAM."
        )

    data_dict, img_keys = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video, img_keys)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
