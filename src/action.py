from __future__ import annotations

import torch

from copy import deepcopy
from dataclasses import dataclass

from src.config import Config


@dataclass
class Action:
    config: Config
    possible_actions: list
    action_idx: int | None
    prob: torch.tensor | None

    @staticmethod
    def create_from(
        config: Config, action_idx: int, prob: torch.tesnor = torch.tensor(1.0)
    ) -> Action:
        assert config is not None
        assert 0 <= action_idx and action_idx < len(config.possible_actions)
        assert prob is not None
        assert torch.tensor(0.0) <= prob and prob <= torch.tensor(1.0)

        return Action(
            config=config,
            possible_actions=config.possible_actions,
            action_idx=action_idx,
            prob=prob,
        )

    def get_udpate(self):
        return self.possible_actions[self.action_idx]

    def take_action(self, action_idx: int, prob: torch.tensor):
        assert 0 <= action_idx and action_idx < len(self.possible_actions)
        assert prob is not None
        assert torch.tensor(0.0) <= prob and prob <= torch.tensor(1.0)

        self.action_idx = action_idx
        self.prob = prob
        return self.get_udpate()

    def reward(self) -> torch.tesnor:
        dx, dy = self.possible_actions[self.action_idx]

        # RF:
        #  1. x-direction: desire move left->right
        #  2. y-direction: not desire any move
        # return torch.tensor(dx - abs(dy))
        return torch.tensor(dx - abs(dy))
