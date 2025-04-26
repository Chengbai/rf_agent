from __future__ import annotations

import matplotlib

from dataclasses import dataclass
from src.action import Action
from src.agent import Agent
from src.config import Config
from src.state import State


@dataclass
class World:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

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

        action_reward = action
        reward = max(0, state.x + action)
        return reward

    def viz(self, ax: matplotlib.axes._axes.Axes, agent: Agent, config: Config):
        assert ax is not None
        assert agent is not None

        x0 = agent.state.x
        y0 = agent.state.y

        xa = []
        ya = []
        for idx, action in enumerate(agent.action_history):
            dx, dy = action.get_udpate()
            xa.append(x0 + dx)
            ya.append(y0 + dy)
            x0 += dx
            y0 += dy
            ax.annotate(f"{idx}", xy=(x0, y0), xycoords="data", fontsize=12)
        ax.plot(xa, ya, marker="o", linestyle="--", color="red")

        # ax.set(
        #     xlim=[config.world_min_x, config.world_max_x],
        #     ylim=[config.world_min_y, config.world_max_y],
        # )
        ax.grid(which="major", axis="y")
