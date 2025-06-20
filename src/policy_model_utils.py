import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.action import Action
from src.config import Config
from src.episode import Episode
from src.episode_dataset import EpisodeRLDataset
from src.policy.policy_base import PolicyBaseModel
from src.policy_factory import PolicyMode, PolicyFactory
from src.reward_model import RewardModel
from src.rl_data_record import RLDataRecord
from src.utils import get_color, top_k_sampling


def save_policy_model(policy: PolicyBaseModel):
    assert policy is not None
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_path = f"rf_model_policy_{now_str}.pt"
    print(f"model_path: {model_path}")
    torch.save(policy.state_dict(), model_path)
    print(f"Save policy model to: {model_path}")


def load_policy_model(config: Config, policy_model_path: str) -> PolicyBaseModel:
    assert policy_model_path
    assert Path(policy_model_path).exists()
    policy = PolicyFactory.create(policy_mode=PolicyMode.LINEAR_MODEL, config=config)
    policy.load_state_dict(torch.load(policy_model_path))
    return policy


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


def inference_and_plot_pre_train_policy(
    config: Config,
    dataset: EpisodeRLDataset,
    dataloader: DataLoader,
    policy: PolicyBaseModel,
    steps: int,
):
    assert config is not None
    assert dataset is not None
    assert dataloader is not None
    assert policy is not None
    assert len(dataloader) > 0

    with tqdm(dataloader, desc=f"{dataset.split}") as t:
        cur_batch_episode_idx = None
        batch_rl_data_record = None
        for batch_idx, batch_data in enumerate(t):
            # step = epoch * len(train_dataloader) + batch_idx
            if batch_idx >= 10:
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

            for idx, episode in enumerate(target_episodes):
                if idx == 0:  # viz the 1st batch 1st item
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
                        batch_fov[idx][0].cpu(),
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
                    plt.show()

                episode.reset()

            t.set_postfix(
                {
                    "split": dataset.split,
                    "batch_idx": batch_idx,
                    "target_episodes": [e.episode_id for e in target_episodes],
                }
            )
