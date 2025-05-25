from __future__ import annotations

import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision.transforms.functional as F

from dataclasses import dataclass
from matplotlib.colors import ListedColormap

from src.action import Action
from src.agent import Agent
from src.config import Config
from src.state import State

# Create a custom colormap for the cells
# cmap = ListedColormap(Config.ENCODE_COLORS)


@dataclass
class World:
    id: str
    config: Config
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    world_board: torch.tensor
    world_board_with_fov_padding: torch.tensor

    @staticmethod
    def create_from_config(id: str, config: Config) -> World:
        assert id
        assert config is not None
        return World(
            id=id,
            config=config,
            x_min=config.world_min_x,
            x_max=config.world_max_x,
            y_min=config.world_min_y,
            y_max=config.world_max_y,
            world_board=torch.zeros(
                size=(config.world_height, config.world_width)
            ),  # Rows x Columns = H x W = Y x X
            world_board_with_fov_padding=None,  # will be initialized in `random_init`
        ).random_init(probability=config.world_block_probability)

    def random_init(self, probability: float = 0.05) -> World:
        self.world_board = torch.tensor(
            np.random.choice(
                [self.config.ENCODE_EMPTY, self.config.ENCODE_BLOCK],
                size=self.config.world_height * self.config.world_width,
                p=[1.0 - probability, probability],
            ),
            dtype=torch.float,
        ).view(self.config.world_height, self.config.world_width)
        self.world_board_with_fov_padding = F.pad(
            self.world_board,
            [self.config.field_of_view_height, self.config.field_of_view_width] * 2,
            self.config.ENCODE_BLOCK,
            padding_mode="constant",
        )
        return self

    def clone(self, idx: int) -> World:
        return World(
            id=f"{self.id}:clone:{idx}",
            config=self.config,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            world_board=self.world_board.clone(),
            world_board_with_fov_padding=self.world_board_with_fov_padding.clone(),
        )

    def get_reward(self, state: State, action: Action):
        assert state is not None
        assert action is not None

        # reward rule: reward = (x2 - x1) - abs(y2 - y1)

        action_reward = action
        reward = max(0, state.x + action)
        return reward

    def fov(self, center_pos: torch.tensor) -> torch.tensor:
        assert center_pos is not None
        assert center_pos.size()[-1] == 2

        # crop a (FOV+1) x (FOV+1)
        """
            Y-axis
            ^
            |      2 x FOV
        (sX-FOV-1, sY+FOV+1)   (sX+FOV+1, sY+FOV+1)
            |------------------|
            |                  |
            |                  |
            |     (sX, sY)     | 2 x FOV
            |                  |
            |                  |
            |------------------|-> X-axis
        (sX-FOV-1, sY-FOV-1)   (sX+FOV, sY-FOV-1)
        """

        top_left_pos = center_pos + torch.tensor(
            [
                -self.config.field_of_view_height - 1,
                -self.config.field_of_view_width - 1,
            ]
        )

        sy = int(top_left_pos[0])  # Rows -> y-axis
        sx = int(top_left_pos[1])  # Columns -> x-axis
        fov_tensor = F.crop(
            self.world_board_with_fov_padding,
            sx
            + self.config.field_of_view_width
            + 1,  # because padding all 4 sides with `field_of_view`
            sy
            + self.config.field_of_view_height
            + 1,  # because padding all 4 sides with `field_of_view`
            2 * self.config.field_of_view_height + 1,
            2 * self.config.field_of_view_width + 1,
        )

        return fov_tensor.clone()

    def viz_fov(self, center_pos: torch.tensor, ax: matplotlib.axes._axes.Axes):
        assert center_pos is not None
        assert ax is not None

        fov = self.fov(center_pos=center_pos)
        ax.pcolormesh(fov, cmap=self.config.CMAP, edgecolors="gray", linewidths=0.5)
        # ax.pcolormesh(fov, cmap=self.config.CMAP, edgecolors="gray", linewidths=0.5)
        # ax.set_xlim(self.config.world_min_x, self.config.world_max_x)
        # ax.set_ylim(self.config.world_min_y, self.config.world_max_y)

    def viz(
        self,
        ax: matplotlib.axes._axes.Axes,
        agent: Agent,
        config: Config,
        color: str = "red",
    ):
        assert ax is not None
        assert agent is not None

        # make a copy for the viz
        viz_world_board = self.world_board.clone().detach()

        x0 = int(agent.start_state.x)
        y0 = int(agent.start_state.y)
        viz_world_board[y0][x0] = self.config.ENCODE_START_POS

        ax.annotate(
            f"start",
            xy=(x0, y0),
            xycoords="data",
            color="black",
            fontsize=12,
        )

        x1 = int(agent.target_state.x)
        y1 = int(agent.target_state.y)
        viz_world_board[y1][x1] = self.config.ENCODE_TARGET_POS
        ax.annotate(
            f"target",
            xy=(x1, y1),
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
            viz_world_board[y0][
                x0
            ] = self.config.ENCODE_START_STEP_IDX  # update the trace
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

        # ax.pcolormesh(viz_world_board, cmap=self.config.CMAP, edgecolors="gray", linewidths=0.5)
        ax.pcolormesh(
            viz_world_board, cmap=self.config.CMAP, edgecolors="gray", linewidths=0.5
        )
        ax.set_xlim(self.config.world_min_x, self.config.world_max_x)
        ax.set_ylim(self.config.world_min_y, self.config.world_max_y)

        world_board_h, world_board_w = viz_world_board.shape
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

    def reset(self):
        pass

    def can_move_to(self, target_position: torch.tensor):
        x, y = target_position
        if (
            x < 0
            or x >= self.config.world_width
            or y < 0
            or y >= self.config.world_height
        ):
            return False
        return self.world_board[y][x] == 0  # H x W
