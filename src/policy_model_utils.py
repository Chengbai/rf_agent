import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from src.action import Action
from src.config import Config
from src.episode import Episode
from src.policy import Policy
from src.reward_model import RewardModel
from src.utils import get_color


def save_policy_model(policy: Policy):
    assert policy is not None
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_path = f"rf_model_policy_{now_str}.pt"
    print(f"model_path: {model_path}")
    torch.save(policy.state_dict(), model_path)
    print(f"Save policy model to: {model_path}")


def load_policy_model(config: Config, policy_model_path: str) -> Policy:
    assert policy_model_path
    assert Path(policy_model_path).exists()
    policy = Policy(config=config)
    policy.load_state_dict(torch.load(policy_model_path))
    return policy


def train_and_plot_policy(
    policy: Policy, config: Config, reward_model: RewardModel, debug: bool = False
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
    policy: Policy,
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
