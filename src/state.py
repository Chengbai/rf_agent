from __future__ import annotations

import torch

from dataclasses import dataclass

from src.action import Action
from src.config import Config


@dataclass
class State:
    config: Config
    id: str
    x: int
    y: int

    @staticmethod
    def create_from(config: Config, id: str, x: int, y: int) -> State:
        assert config is not None
        assert id

        return State(config=config, id=id, x=x, y=y)

    def copy(self) -> State:
        return State(config=self.config, id=f"{self.id}_copy", x=self.x, y=self.y)

    def can_take_action(self, action: Action) -> bool:
        dx, dy = action.get_udpate()
        next_x = self.x + dx
        next_y = self.y + dy

        if (
            next_x <= self.config.world_min_x
            or next_x >= self.config.world_max_x
            or next_y <= self.config.world_min_y
            or next_y >= self.config.world_max_y
        ):
            return False

        return True

    def take_action(self, action: Action) -> State:
        assert action is not None

        assert self.can_take_action(action=action)

        dx, dy = action.get_udpate()
        self.x = min(
            max(self.x + dx, self.config.world_min_x), self.config.world_max_x - 1
        )
        self.y = min(
            max(self.y + dy, self.config.world_min_y), self.config.world_max_y - 1
        )
        return self

    def normalized_position(self) -> torch.tensor:
        return torch.tensor(
            [
                self.x / (self.config.world_max_x - self.config.world_min_x),
                self.y / (self.config.world_max_y - self.config.world_min_y),
            ]
        )

    def position(self) -> torch.tensor:
        return torch.tensor([self.x, self.y])
