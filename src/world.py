from __future__ import annotations

from dataclasses import dataclass
from src.action import Action
from src.agent import Agent
from src.config import Config
from src.state import State


@dataclass
class World:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @staticmethod
    def create_from_config(config: Config) -> World:
        assert config is not None
        return World(
            x_min=config.world_min_x,
            x_max=config.world_max_x,
            y_min=config.world_min_y,
            y_max=config.world_max_y,
        )

    def get_reward(self, state: State, action: Action):
        assert state is not None
        assert action is not None

        # reward rule: reward = (x2 - x1) - abs(y2 - y1)

    def viz(self, agent: Agent):
        pass
