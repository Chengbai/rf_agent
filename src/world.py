from __future__ import annotations

import matplotlib
import numpy as np
import torch

from dataclasses import dataclass
from matplotlib.colors import ListedColormap

from src.action import Action
from src.agent import Agent
from src.config import Config
from src.state import State

# Create a custom colormap for the cells
colors = ["white", "black", "red", "blue"]
cmap = ListedColormap(colors)


@dataclass
class World:
    config: Config
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    world_board: torch.tensor

    @staticmethod
    def create_from_config(config: Config) -> World:
        assert config is not None
        return World(
            config=config,
            x_min=config.world_min_x,
            x_max=config.world_max_x,
            y_min=config.world_min_y,
            y_max=config.world_max_y,
            world_board=torch.zeros(
                size=(config.world_height, config.world_width)
            ),  # Rows x Columns = H x W = Y x X
        ).random_init(probability=0.01)

    def random_init(self, probability: float = 0.05) -> World:
        for h in range(self.config.world_height):
            for w in range(self.config.world_width):
                self.world_board[h][w] = torch.tensor(
                    np.random.choice([0, 1], size=1, p=[1.0 - probability, probability])
                )
        return self

    def get_reward(self, state: State, action: Action):
        assert state is not None
        assert action is not None

        # reward rule: reward = (x2 - x1) - abs(y2 - y1)

        action_reward = action
        reward = max(0, state.x + action)
        return reward

    def viz(
        self,
        ax: matplotlib.axes._axes.Axes,
        agent: Agent,
        config: Config,
        color: str = "red",
    ):
        assert ax is not None
        assert agent is not None

        # debug
        # self.world_board[0][0] = 2

        x0 = int(agent.init_state.x)
        y0 = int(agent.init_state.y)
        self.world_board[y0][x0] = 2

        ax.annotate(
            f"start",
            xy=(x0, y0),
            xycoords="data",
            color="black",
            fontsize=12,
        )

        xa = [x0]
        ya = [y0]
        for idx, action in enumerate(agent.action_history):
            dx, dy = action.get_udpate()
            dx = int(dx)
            dy = int(dy)
            xa.append(x0 + dx)
            ya.append(y0 + dy)
            # print(f"step: {idx}, x0: {x0}, y0: {y0} dx: {dx}, dy: {dy}")
            x0 += dx
            y0 += dy
            self.world_board[y0][x0] = 3  # update the trace
            # ax.annotate(
            #     f"{idx}(P{action.prob.item()*100:.2f}%)",
            #     xy=(y0, x0),
            #     xycoords="data",
            #     fontsize=5,
            # )
        # Sort data based on x-values
        # xa = np.array(xa)
        # ya = np.array(ya)
        # sorted_indices = np.argsort(xa)
        # xa_sorted = xa[sorted_indices]
        # ya_sorted = ya[sorted_indices]
        # ax.plot(xa_sorted, ya_sorted, marker="o", linestyle="--", color=color)
        # print(f"xa_sorted: {xa_sorted}")
        # print(f"ya_sorted: {ya_sorted}")

        ax.pcolormesh(self.world_board, cmap=cmap, edgecolors="gray", linewidths=0.5)
        ax.set_xlim(self.config.world_min_x, self.config.world_max_x)
        ax.set_ylim(self.config.world_min_y, self.config.world_max_y)

        world_board_h, world_board_w = self.world_board.shape
        ax.set_xticks(list(range(0, world_board_w, 5)))  # Set x-ticks at 1, 2, and 3
        ax.set_yticks(list(range(0, world_board_h, 5)))  # Set y-ticks at 4, 8, and 12
        ax.grid(True)  # Show gridlines at the set tick positions

        # ax.set_xticklabels([])
        # ax.set_yticklabels([])

        # ax.set(
        #     xlim=[config.world_min_x, config.world_max_x],
        #     ylim=[config.world_min_y, config.world_max_y],
        # )
        ax.grid(which="major", axis="y")
