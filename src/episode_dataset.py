from __future__ import annotations

import torch
from torch.utils.data import Dataset

from concurrent.futures import ThreadPoolExecutor
from random import random
import time

from src.action import Action
from src.agent import Agent
from src.config import Config
from src.episode import Episode
from src.state import State
from src.world import World
import src.utils as utils


class EpisodeDataset(Dataset):
    def __init__(self, config: Config, split: str):
        super().__init__()
        assert config is not None
        assert split
        assert split in ["TRAIN", "TEST", "EVAL"]
        self.config: Config = config
        self.split: str = split
        self.episodes: list[Episode] = []

        if split == "TRAIN":
            dataset_length = config.train_dataset_length
        elif split == "TEST":
            dataset_length = config.test_dataset_length
        else:
            assert split == "EVAL"
            dataset_length = config.eval_dataset_length

        # Dataset pattern:
        #   - for group size 3
        #   - [E0, E0C0, E0C1, E1, E1C0, E1C2, ...]
        for episode_idx in range(dataset_length):
            # Create the 1st episode in the group
            episode = Episode.new(episode_id=f"episode_{split}_{episode_idx}")
            self.episodes.append(episode)

            # Clone G-1 times
            self.episodes.extend(episode.clone(repeat=config.episode_group_size - 1))

        assert len(self.episodes) == dataset_length * self.config.episode_group_size

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        # print(f"episode_idx: {index}")
        target_episode = self.episodes[index % len(self.episodes)]
        assert target_episode is not None

        return {
            "episode_idx": index,
            "agent_start_pos": target_episode.agent.start_state.position(),
            "agent_target_pos": target_episode.agent.target_state.position(),
            "agent_current_pos": target_episode.agent.current_state.position(),
            # "world_id": target_episode.world.world_id,
            # "position": target_episode.agent.current_state.position(),
            # "normalized_position": target_episode.agent.current_state.normalized_position(),
            "fov": target_episode.fov(target_episode.agent.current_state.position()),
            "fov_block_mask": target_episode.fov(
                target_episode.agent.current_state.position()
            )
            == self.config.ENCODE_BLOCK,
        }

    def get_episode(self, index: int) -> Episode:
        return self.episodes[index % len(self.episodes)]

    def update_step(
        self,
        batch_episode_indices: list[int],
        batch_action_idx: torch.Tensor,
        batch_logit_prob: torch.Tensor,
        batch_top_k_prob: torch.Tensor,
    ):
        assert batch_episode_indices is not None
        assert batch_action_idx is not None
        assert batch_logit_prob is not None
        assert batch_top_k_prob is not None
        assert (
            len(batch_episode_indices)
            == batch_action_idx.shape[0]
            == batch_logit_prob.shape[0]
            == batch_top_k_prob.shape[0]
        )

        # Batch update the episodes
        def _update_episode(
            action_idx: int,
            prob: torch.Tensor,
            episode_id: int,
        ):
            episode: Episode = self.get_episode(episode_id)
            assert episode is not None

            episode.take_action(
                action=Action.create_from(
                    config=self.config,
                    action_idx=action_idx,
                    prob=prob,
                ),
            )

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for item_idx, episode_idx in enumerate(batch_episode_indices):
                action_idx = batch_action_idx[item_idx][0]
                prob = batch_logit_prob[item_idx]
                future = executor.submit(_update_episode, action_idx, prob, episode_idx)

            for future in futures:
                future.result()

        # for item_idx, episode_idx in enumerate(batch_episode_indices):
        #     episode: Episode = self.get_episode(episode_idx)
        #     assert episode is not None

        #     episode.take_action(
        #         action=Action.create_from(
        #             config=self.config,
        #             action_idx=batch_action_idx[item_idx][0],
        #             prob=batch_logit_prob[item_idx],
        #         ),
        #     )

    def get_episods(self, batch_episode_indices: list[int]) -> list[Episode]:
        target_episodes = []
        for episode_idx in batch_episode_indices:
            episode: Episode = self.get_episode(episode_idx)
            assert episode is not None
            target_episodes.append(episode)

        return target_episodes
