"""
This module is created on 2025
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


from src.config import Config


@dataclass
class Action:
    """Define the action taken in an episode step."""

    config: Config
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
            action_idx=action_idx,
            prob=prob,
        )

    def clone(self, idx: int) -> Action:
        return Action(
            config=self.config,
            action_idx=self.action_idx,
            prob=self.prob.clone() if self.prob is not None else None,
        )

    @property
    def possible_actions(self):
        return self.config.possible_actions

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
