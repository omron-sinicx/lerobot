"""
This script is used to train ACT policy on the dataset.
"""

from pathlib import Path

import torch
import yaml

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from copy import deepcopy

# Create a directory to store the training checkpoint.
output_directory = Path("outputs/train/act_contactile_300_deltas_act")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 5000
device = torch.device("cuda")
log_freq = 250

# Set up the dataset.
with open("configs/act.yaml") as f:
    config_dict = yaml.safe_load(f)

# 除外するキー
config_for_class = {k: v for k, v in config_dict.items() if k not in ["dataset_root", "repo_id"]}

cfg = ACTConfig(**config_for_class)
cfg = deepcopy(cfg)

# Read dataset path and repo_id from config
DATASET_ROOT = config_dict.get("dataset_root", None)
REPO_ID = config_dict.get("repo_id", None)

delta_timestamps = {
    "observation.qpos": [0.0],
    "observation.ft": [0.0],
    "observation.eef.position": [0.0],
    "observation.eef.rotation_ortho6": [0.0],
    "observation.vive_tracker_pose": [0.0],
    "observation.contactile": [0.0],
    "action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
}

dataset = LeRobotDataset(root=DATASET_ROOT, repo_id=REPO_ID, delta_timestamps=delta_timestamps)
# dataset = LeRobotDataset(root=DATASET_ROOT, repo_id=REPO_ID)

# Debug: print available stats keys
print("Available dataset.stats keys:", dataset.stats.keys())

# Set up the the policy.
# Policies are initialized with a configuration class, in this case `DiffusionConfig`.
# For this example, no arguments need to be passed because the defaults are set up for PushT.
# If you're doing something different, you will likely need to change at least some of the defaults.
policy = ACTPolicy(cfg, dataset_stats=dataset.stats)
policy.train()
policy.to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

# Create dataloader for offline training.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
)

# Run training loop.
step = 0
done = False
while not done:
    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output_dict = policy.forward(batch)
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

# Save a policy checkpoint.
policy.save_pretrained(output_directory)

