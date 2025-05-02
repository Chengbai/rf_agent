from __future__ import annotations

import torch
from dataclasses import dataclass

from src.action import Action
from src.config import Config


@dataclass
class State:
    config: Config
    id: str
    x: float
    y: float

    @staticmethod
    def create_from(config: Config, id: str, x: float, y: float) -> State:
        assert config is not None
        assert id

        return State(config=config, id=id, x=x, y=y)

    def take_action(self, action: Action) -> State:
        assert action is not None

        dx, dy = action.get_udpate()
        self.x = min(max(self.x + dx, self.config.world_min_x), self.config.world_max_x)
        self.y = min(max(self.y + dy, self.config.world_min_y), self.config.world_max_y)
        return self

    def to_tensor(self) -> torch.tensor:
        return torch.tensor(
            [
                self.x / (self.config.world_max_x - self.config.world_min_x),
                self.y / (self.config.world_max_y - self.config.world_min_y),
            ]
        ).to(self.config.device)
