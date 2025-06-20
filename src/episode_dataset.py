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

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def get_episode(self, index: int) -> Episode:
        raise NotImplementedError()

    def update_step(
        self,
        batch_episode_indices: list[int],
        batch_action_idx: torch.Tensor,
        batch_logit_prob: torch.Tensor,
        batch_top_k_prob: torch.Tensor,
    ):
        raise NotImplementedError()

    def get_episods(self, batch_episode_indices: list[int]) -> list[Episode]:
        raise NotImplementedError()


class EpisodeSupervisedDataset(EpisodeDataset):
    """This class is for providing the supervised next position prediction data."""

    def __init__(self, config, split):
        super().__init__(config, split)

        if split == "TRAIN":
            dataset_length = config.train_dataset_length
        elif split == "TEST":
            dataset_length = config.test_dataset_length
        else:
            assert split == "EVAL"
            dataset_length = config.eval_dataset_length

        # Dataset pattern:
        for episode_idx in range(dataset_length):
            # Create the 1st episode in the group
            episode = Episode.new(episode_id=f"episode_{split}_{episode_idx}")
            self.episodes.append(episode)

        assert len(self.episodes) == dataset_length

        # Best path could be NOT exist. For this case, backfill a new one
        bad_episodes = [
            episode for episode in self.episodes if episode.best_path is None
        ]
        backfill_episodes = []
        for episode in bad_episodes:
            while True:
                new_episode = Episode.new(episode_id=episode.episode_id)
                if new_episode.best_path is not None:
                    backfill_episodes.append(new_episode)
                    break
        self.episodes = [
            episode for episode in self.episodes if episode.best_path is not None
        ] + backfill_episodes

        #  - Each episode has it's own `best path`, and the length of the it could be vary
        #  - following dict diagram shows an example for the first 2 episodes, one have 2 and another one has 3 length:
        # {
        #     0:  ep1.best_path[0],
        #     1:  ep1.best_path[1],
        #     2:  ep2.best_path[0],
        #     3:  ep2.best_path[1],
        #     4:  ep2.best_path[2]
        #     ...
        # }

        self.index_to_episode_map = {}
        cur_idx = 0
        for episode in self.episodes:
            for _ in range(len(episode.best_path)):
                self.index_to_episode_map[cur_idx] = episode
                cur_idx += 1

    def __len__(self):
        # Each episode has a `best path` which records a list of the `best next position`.
        return len(self.index_to_episode_map)

    def __getitem__(self, index: int):
        # print(f"episode_idx: {index}")
        index = index % len(self.index_to_episode_map)
        target_episode = self.get_episode(index=index)

        #          ep1.best_path[0] | ep1.best_path[1] | ep1.best_path[2] | ...
        # offset:  0                | 1                | 2
        offset_from_episode_start_index = self._offset_from_episode_start_index(
            cur_index=index, cur_episode=target_episode
        )
        best_next_pos: torch.Tensor = target_episode.best_path[
            offset_from_episode_start_index
        ]
        agent_current_pos: torch.Tensor = (
            target_episode.agent.start_state.position()
            if offset_from_episode_start_index == 0
            else target_episode.best_path[offset_from_episode_start_index - 1]
        )
        best_next_action = torch.tensor(
            self.config.action_to_idx[
                tuple((best_next_pos - agent_current_pos).to(torch.int).tolist())
            ]
        )

        return {
            "episode_idx": index,
            "agent_start_pos": target_episode.agent.start_state.position(),
            "agent_target_pos": target_episode.agent.target_state.position(),
            "agent_current_pos": agent_current_pos,
            "best_next_pos": best_next_pos,
            "best_next_action": best_next_action,
            # "world_id": target_episode.world.world_id,
            # "position": target_episode.agent.current_state.position(),
            # "normalized_position": target_episode.agent.current_state.normalized_position(),
            "fov": target_episode.fov(target_episode.agent.current_state.position())[
                None, ...
            ],  # C(=1) x H x W
            "fov_block_mask": target_episode.fov(
                target_episode.agent.current_state.position()
            )
            == self.config.ENCODE_BLOCK,
        }

    def _offset_from_episode_start_index(self, cur_index: int, cur_episode: Episode):
        assert cur_index >= 0
        index = cur_index
        # Each episode's best_path is limited length. We do quick linear search here is ok. Consider O(1)
        while index >= 0:
            if self.index_to_episode_map[index] != cur_episode:
                break
            index -= 1

        offset = cur_index - index - 1  # off 1
        return offset

    def get_episode(self, index: int) -> Episode:
        index = index % len(self.index_to_episode_map)
        target_episode = self.index_to_episode_map[index]
        assert target_episode is not None
        return target_episode

    def get_episods(self, batch_episode_indices: list[int]) -> list[Episode]:
        target_episodes = []
        for episode_idx in batch_episode_indices:
            episode: Episode = self.get_episode(episode_idx)
            assert episode is not None
            target_episodes.append(episode)

        return target_episodes


class EpisodeRLDataset(EpisodeDataset):
    """This class if for providing the RL data."""

    def __init__(self, config, split):
        super().__init__(config, split)

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
            # "best_path": target_episode.best_path,
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
