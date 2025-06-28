import cv2
import io
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import torchvision


from src.action import Action
from src.config import Config
from src.episode import Episode
from src.episode_dataset import EpisodeDataset, EpisodeRLDataset
from src.policy.policy_base import PolicyBaseModel
from src.policy_factory import PolicyMode, PolicyFactory
from src.reward_model import RewardModel
from src.rl_data_record import RLDataRecord
from src.train_stage import TrainStage
from src.utils import get_color, top_k_sampling, clean_data_cache


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def save_checkpoint(
    run_id: str,
    train_stage: TrainStage,
    model: PolicyBaseModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    lr_sched: torch.optim.lr_scheduler.LRScheduler,
):
    assert model is not None

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_sched.state_dict(),
    }

    checkpoint_path = f"rf_model_policy_{train_stage.name}_{run_id}_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Save checkpoint to: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    config: Config,
    policy_mode: PolicyMode = PolicyMode.TRANSFORMER_WITH_LATE_POSITION_FUSION,
) -> tuple[
    PolicyBaseModel, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    assert Path(checkpoint_path).exists()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load policy
    policy = PolicyFactory.create(policy_mode=policy_mode, config=config)
    policy.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer
    # print(f'optimizer: {checkpoint["optimizer_state_dict"]}')
    optimizer = torch.optim.AdamW(policy.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # LR scheduler
    scheduler_state_dict = checkpoint["scheduler_state_dict"]
    # print(f"lr_scheduler: {scheduler_state_dict}")
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_state_dict["T_max"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return policy, optimizer, lr_scheduler


def load_policy_model(
    config: Config, policy_mode: PolicyMode, policy_model_path: str
) -> PolicyBaseModel:
    assert policy_model_path
    assert Path(policy_model_path).exists()
    policy = PolicyFactory.create(policy_mode=policy_mode, config=config)
    policy.load_state_dict(torch.load(policy_model_path))
    return policy


def get_model_size(model: torch.nn.Module):
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Calculate model size in bytes (assuming float32)
    total_bytes = total_params * 4  # Assuming float32
    print(f"Model size (bytes): {total_bytes}")

    # Calculate model size in MB
    total_mb = total_bytes / (1024 * 1024)
    print(f"Model size (MB): {total_mb}")


def train_and_plot_policy(
    policy: PolicyBaseModel,
    config: Config,
    reward_model: RewardModel,
    debug: bool = False,
):
    episode = Episode.new(id="train")
    print(f"start: {episode.agent.current_state}")
    episode.train(steps=20, policy=policy, debug=debug)
    print(f"start2: {episode.agent.current_state}")

    fig = plt.figure(figsize=config.figure_size)
    ax = fig.add_subplot(1, 1, 1)
    episode.viz(ax=ax, reward_model=reward_model, color=get_color(0))
    plt.show()

    return episode


def inference_and_plot_policy(
    policy: PolicyBaseModel,
    config: Config,
    reward_model: RewardModel,
    steps: int = 20,
    episode: Episode = None,
    debug: bool = False,
):
    _, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=config.triple_figure_size,
    )
    if episode is None:
        episode = Episode.new(episode_id="inference")
    episode.viz_fov(ax=axes[0])
    axes[0].set_title(f"{episode.episode_id}: Initial state")

    # print(f"start state: {episode.agent.start_state}")
    # print(f"target state: {episode.agent.target_state}")
    episode.inference_steps_by_policy(steps=steps, policy=policy, debug=debug)
    # print(f"end state: {episode.agent.current_state}")

    episode.viz(ax=axes[1], reward_model=reward_model, color=get_color(0))

    episode.viz_fov(ax=axes[2])
    axes[2].set_title(f"{episode.episode_id}: Final state")
    plt.show()

    return episode


def inference_and_plot_policy_v2(
    config: Config,
    dataset: EpisodeRLDataset,
    dataloader: DataLoader,
    policy: PolicyBaseModel,
    reward_model: RewardModel,
    top_k: int = 2,
):
    assert config is not None
    assert dataset is not None
    assert dataloader is not None
    assert policy is not None
    assert reward_model is not None
    assert len(dataloader) > 0

    with tqdm(dataloader, desc=f"{dataset.split}") as t:
        cur_batch_episode_idx = None
        batch_rl_data_record = None
        for batch_idx, batch_data_items in enumerate(t):
            if batch_idx > 10 * config.episode_steps:
                break

            # step = epoch * len(train_dataloader) + batch_idx
            if batch_rl_data_record is None:
                batch_rl_data_record = RLDataRecord(
                    config=config, batch_data_items=batch_data_items
                )

            cur_batch_episode_idx = batch_data_items["episode_idx"]

            batch_logits = policy.execute_1_step(
                batch_rl_data_record=batch_rl_data_record
            )
            batch_action_idx, batch_logit_prob, batch_top_k_prob = top_k_sampling(
                logits=batch_logits, k=top_k
            )
            # print(
            #     f"batch_action_idx: {batch_action_idx}, batch_logit_prob: {batch_logit_prob}, batch_top_k_prob: {batch_top_k_prob}"
            # )
            batch_rl_data_record.update_step(
                batch_action_idx=batch_action_idx,
                batch_logit_prob=batch_logit_prob,
                batch_top_k_prob=batch_top_k_prob,
                step=batch_idx,
                debug=False,
            )

            is_episode_step_done = (batch_idx + 1) % config.episode_steps == 0
            if is_episode_step_done:
                assert (
                    cur_batch_episode_idx
                    == batch_rl_data_record.current_batch_episode_idx
                )
                target_episodes = dataset.get_episods(
                    batch_episode_indices=batch_rl_data_record.current_batch_episode_idx
                )
                for idx, episode in enumerate(target_episodes):
                    if idx == 0:  # viz the 1st batch 1st item
                        # only viz the 1st episode
                        # avoid too much data
                        fig, axes = plt.subplots(
                            nrows=1,
                            ncols=3,
                            figsize=config.triple_figure_size,
                        )
                        episode.viz(
                            ax=axes[0],
                            reward_model=reward_model,
                            color=get_color(0),
                        )
                        episode.viz_fov(
                            ax=axes[1],
                        )
                        batch_rl_data_record.viz_fov(
                            ax=axes[2], idx=idx, reward_model=reward_model
                        )
                        plt.show()

                    episode.reset()

                t.set_postfix(
                    {
                        "split": dataset.split,
                        "batch_idx": batch_idx,
                        "is_episode_step_done": is_episode_step_done,
                        "target_episodes": [e.episode_id for e in target_episodes],
                        "current_batch_episode_idx": batch_rl_data_record.current_batch_episode_idx,
                    }
                )
                batch_rl_data_record = None


def save_episode_to_img(
    episode: Episode, episode_fov: torch.Tensor, episode_img_path: str, config: Config
):
    start_x = int(episode.agent.start_state.x)
    start_y = int(episode.agent.start_state.y)
    target_x = int(episode.agent.target_state.x)
    target_y = int(episode.agent.target_state.y)

    # only viz the 1st episode
    # avoid too much data
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=config.triple_figure_size,
    )
    episode.viz_fov(
        ax=axes[0],
    )
    episode.viz_optimal_path(
        ax=axes[1],
    )
    axes[2].pcolormesh(
        episode_fov,
        cmap=config.CMAP,
        edgecolors="gray",
        linewidths=0.5,
    )
    for ax in axes:
        ax.annotate(
            f"start",
            xy=(start_x, start_y),
            xycoords="data",
            color="black",
            fontsize=12,
        )
        ax.annotate(
            f"target",
            xy=(target_x, target_y),
            xycoords="data",
            color="black",
            fontsize=12,
        )

    # Create an in-memory binary stream
    buffer = io.BytesIO()

    # Save the figure to the buffer
    fig.savefig(buffer, format="png")  # Specify the format (e.g., 'png')

    # The image data is now in the buffer
    buffer.seek(
        0
    )  # Reset the buffer's position to the beginning if you want to read from it

    image_data = buffer.getvalue()
    with open(episode_img_path, "wb") as f:
        f.write(image_data)
        # print(f"img: {episode_img_path}")

    plt.close()
    return episode_img_path


def save_imgs_to_video(episode_img_group: list[str], episode_video_path: str):
    # print(f"episode_img_group: {episode_img_group}")
    # Read the first image to get the size
    frame = cv2.imread(episode_img_group[0])
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for MP4 format
    video = cv2.VideoWriter(episode_video_path, fourcc, 2, (width, height))

    # Add images to the video
    for image_path in episode_img_group:
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer
    video.release()
    cv2.destroyAllWindows()


def inference_and_plot_pre_train_policy(
    config: Config,
    dataset: EpisodeDataset,
    dataloader: DataLoader,
    policy: PolicyBaseModel,
    steps: int,
) -> list[str]:
    assert config is not None
    assert dataset is not None
    assert dataloader is not None
    assert policy is not None
    assert len(dataloader) > 0

    episode_videos = []
    clean_data_cache(config=config)
    with tqdm(dataloader, desc=f"{dataset.split}") as t:
        cur_batch_episode_idx = None
        batch_rl_data_record = None
        for batch_idx, batch_data in enumerate(t):
            # step = epoch * len(train_dataloader) + batch_idx
            if batch_idx >= 3:
                # only viz the first 4 batches
                break

            batch_fov: torch.Tensor = batch_data["fov"]
            batch_episode_idx: list = batch_data["episode_idx"]
            batch_cur_position: torch.Tensor = batch_data["agent_current_pos"]
            batch_target_position: torch.Tensor = batch_data["agent_target_pos"]
            batch_best_next_pos: torch.Tensor = batch_data["best_next_pos"]
            batch_best_next_action: torch.Tensor = batch_data["best_next_action"]

            batch_origin_cur_position = batch_cur_position.clone()
            batch_origin_batch_fov = batch_fov.clone()
            for step in range(steps):
                batch_logits = policy(
                    batch_fov=batch_origin_batch_fov,
                    batch_cur_position=batch_cur_position,
                    batch_target_position=batch_target_position,
                )

                batch_action_idx, batch_logit_prob, batch_top_k_prob = top_k_sampling(
                    logits=batch_logits, k=1
                )

                # Get the action update
                batch_actions: torch.Tensor = config.possible_actions[
                    batch_action_idx.squeeze(dim=1)
                ]
                # print(f"batch_actions: {batch_actions}")
                batch_agent_next_pos = batch_cur_position + batch_actions
                batch_agent_next_pos[:, 0] = torch.clamp(
                    batch_agent_next_pos[:, 0],
                    min=config.world_min_x,
                    max=config.world_max_x - 1,
                )

                batch_agent_next_pos[:, 1] = torch.clamp(
                    batch_agent_next_pos[:, 1],
                    min=config.world_min_y,
                    max=config.world_max_y - 1,
                )

                B, _ = batch_agent_next_pos.size()
                x_indices = batch_agent_next_pos[:, 0].to(torch.int)
                y_indices = batch_agent_next_pos[:, 1].to(torch.int)
                blocked_pos_mask = (
                    batch_fov[torch.arange(B), 0, y_indices, x_indices]
                    == config.ENCODE_BLOCK  # Row - Y-axis, Col - X-axis
                )
                # Action overwrite:
                #  - cannot move onto the BLOCK
                #  - if already at the TARGET position, no more move
                batch_actions[blocked_pos_mask] = torch.tensor(
                    [0, 0], device=config.device
                )

                # batch_actions[self.batch_at_target_position_mask] = torch.tensor([0, 0])
                batch_cur_position += batch_actions
                batch_cur_position[:, 0] = torch.clamp(
                    batch_cur_position[:, 0],
                    min=config.world_min_x,
                    max=config.world_max_x - 1,
                )

                batch_cur_position[:, 1] = torch.clamp(
                    batch_cur_position[:, 1],
                    min=config.world_min_y,
                    max=config.world_max_y - 1,
                )

                # Update the fov
                x_indices = batch_cur_position[:, 0].to(torch.int)
                y_indices = batch_cur_position[:, 1].to(torch.int)
                batch_fov[torch.arange(B), 0, y_indices, x_indices] = (
                    config.ENCODE_START_STEP_IDX + step  # Row - Y-axis, Col - X-axis
                )

            # Update the episode voz + each step prediction
            target_episodes = dataset.get_episods(
                batch_episode_indices=batch_episode_idx
            )

            episode_img_group = []
            last_episode_id = None
            for idx, episode in enumerate(target_episodes):
                if last_episode_id != episode.episode_id and len(episode_img_group) > 0:
                    episode_video_path = f"{config.mp4_folder}{last_episode_id}_{len(episode_img_group)}.mp4"
                    save_imgs_to_video(episode_img_group, episode_video_path)
                    episode_videos.append(episode_video_path)
                    episode_img_group = []

                episode_img_path = f"{config.mp4_folder}{episode.episode_id}_{idx}.png"
                episode_img_path = save_episode_to_img(
                    episode=episode,
                    episode_fov=batch_fov[idx][0].cpu(),
                    episode_img_path=episode_img_path,
                    config=config,
                )
                episode_img_group.append(episode_img_path)
                last_episode_id = episode.episode_id

                episode.reset()

            t.set_postfix(
                {
                    "split": dataset.split,
                    "batch_idx": batch_idx,
                    "target_episodes": [e.episode_id for e in target_episodes],
                }
            )
    return episode_videos


def add_video_to_tensorboard(
    video_path: str, writer: SummaryWriter, tag: str, global_step: int
):
    assert Path(video_path).exists()
    assert writer is not None
    # frames: (T x H x W x C)
    # metadata: {'video_fps': 2.0}
    frames, _, metadata = torchvision.io.read_video(video_path)
    fps = metadata.get("video_fps", 24)

    # vid_tensor: (N,T,C,H,W)
    vid_tensor = torch.permute(frames, (0, 3, 1, 2))  # (T, C, H, W)
    vid_tensor = vid_tensor[None, ...]  # (1 x T x C x H x W)

    writer.add_video(
        tag=tag, vid_tensor=vid_tensor, global_step=global_step, fps=fps, walltime=None
    )
