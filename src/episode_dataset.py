import torch
from torch.utils.data import Dataset

from random import random

from src.action import Action
from src.agent import Agent
from src.config import Config
from src.episode import Episode
from src.policy import Policy
from src.state import State
from src.world import World
import src.utils as utils


class EpisodeDataset(Dataset):
    def __init__(self, config: Config, split: str):
        super().__init__()
        assert config is not None
        assert split
        assert split in ["TRAIN", "TEST"]
        self.config: Config = config
        self.split: str = split
        self.episodes: list[Episode] = []

        for episode_idx in range(
            config.train_dataset_length
            if split == "TRAIN"
            else config.test_dataset_length
        ):
            episode = Episode.new(id=f"dataitem_{episode_idx}")
            self.episodes.append(episode)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        target_episode = self.episodes[index % len(self.episodes)]
        assert target_episode is not None

        return {
            "episode_idx": index,
            "agent_start_pos": target_episode.agent.start_state.position(),
            "agent_target_pos": target_episode.agent.target_state.position(),
            "agent_current_pos": target_episode.agent.current_state.position(),
            # "world_id": target_episode.world.id,
            # "position": target_episode.agent.current_state.position(),
            # "normalized_position": target_episode.agent.current_state.normalized_position(),
            "fov": target_episode.world.fov(
                target_episode.agent.current_state.position()
            ),
        }

    def get_episode(self, index: int) -> Episode:
        return self.episodes[index % len(self.episodes)]

    def update_step(
        self,
        batch_episode_idices: list[int],
        batch_action_idx: torch.tensor,
        batch_logit_prob: torch.tensor,
        batch_top_k_prob: torch.tensor,
    ):
        assert batch_episode_idices is not None
        assert batch_action_idx is not None
        assert batch_logit_prob is not None
        assert batch_top_k_prob is not None
        assert (
            batch_episode_idices.shape[0]
            == batch_action_idx.shape[0]
            == batch_logit_prob.shape[0]
            == batch_top_k_prob.shape[0]
        )
        for item_idx, episode_idx in enumerate(batch_episode_idices):
            episode: Episode = self.episodes[episode_idx]
            assert episode is not None

            episode.take_action(
                action=Action.create_from(
                    config=self.config,
                    action_idx=batch_action_idx[item_idx][0],
                    prob=batch_logit_prob[item_idx],
                ),
            )

    def get_episods(self, batch_episode_idices: list[int]) -> list[Episode]:
        target_episodes = []
        for episode_idx in batch_episode_idices:
            episode: Episode = self.episodes[episode_idx]
            assert episode is not None
            target_episodes.append(episode)

        return target_episodes
