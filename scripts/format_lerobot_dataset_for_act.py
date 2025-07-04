#!/usr/bin/env python3

"""
This script is used to format the dataset for ACT.

1. Open pth files and concatenate "action.position_cmd" and "action.rotation_cmd" to "action"
2. Save the dataset in lerobot format
3. Compute the dataset statistics

Args:
    --dataset_path: Path to the dataset you want to modify. 
    ** This script updates the dataset directly. Make a copy of the dataset if you want to keep the original. **
"""

import tqdm
import torch
from pathlib import Path
import argparse
import os

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    get_default_encoding,
)
from lerobot.common.datasets.utils import calculate_episode_data_index
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset


def save_lerobot_format(data_dict, dataset_path):
    """
    Save the concatenated dataset in lerobot format.
    """
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    hf_dataset = to_hf_dataset(data_dict, True)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": 50,
        "video": True,
    }
    info["encoding"] = get_default_encoding()
    info["encoding"]["vcodec"] = "libx264"

    # Save to the lerobot format
    dataset_path = Path(dataset_path)
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=dataset_path.name,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=dataset_path / "videos",
    )

    print("Computing dataset statistics")
    stats = compute_stats(lerobot_dataset)
    lerobot_dataset.stats = stats

    hf_dataset = hf_dataset.with_format(None)
    hf_dataset.save_to_disk(str(dataset_path / "train"))

    meta_data_dir = Path(dataset_path) / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)
    print(f"Dataset saved at {dataset_path}")

def concatenate_and_save_episodes(dataset_path):
    """
    Concatenate all episodes and save them in the lerobot dataset format.
    """
    print("Concatenating episodes...")
    num_episodes = len(os.listdir(Path(dataset_path) / "episodes"))
    print(f"Number of episodes: {num_episodes}")
    ep_dicts = []
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        ep_path = Path(dataset_path) / "episodes" / f"episode_{episode_idx}.pth"
        ep_dict = torch.load(ep_path)
        ep_dicts.append(ep_dict)

    # Concatenate episodes
    data_dict = concatenate_episodes(ep_dicts)

    # Save in lerobot format
    print("Saving in lerobot format...")
    save_lerobot_format(data_dict, dataset_path)

def update_pth(dataset_path):
    """
    Update the pth files with the new format.
    - Concatenate "action.position_cmd" and "action.rotation_cmd" to "action"
    """
    print("Updating pth files...")
    num_pth_files = len(list((Path(dataset_path) / "episodes").glob("*.pth")))
    print(f"Number of pth files: {num_pth_files}")
    for pth_file in (Path(dataset_path) / "episodes").glob("*.pth"):
        print(f"Updating {pth_file}...")
        data_dict = torch.load(pth_file)
        data_dict["action"] = torch.cat([data_dict["action.position_cmd"], data_dict["action.rotation_cmd"]], dim=1)
        torch.save(data_dict, pth_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data2/tactile_retrieval/datasets/contactile_300_deltas_act")
    args = parser.parse_args()

    update_pth(args.dataset_path)
    print("Done updating pth files")

    concatenate_and_save_episodes(args.dataset_path)