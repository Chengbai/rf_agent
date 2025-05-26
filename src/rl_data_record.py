"""
This module is created on 2025
"""

import matplotlib
import matplotlib.cm as cm
import torch

from src.config import Config
from src.reward_model import RewardModel


class RLDataRecord:
    """
    - Each train/eval step, it contains a batch of the episodes; each episode is further contains
      a configurable group of clones;
    - Each episode will run configurable steps, at the end, each episode will have a reward
      computed;
    - This data record contains all of the above historical data.
    """

    def __init__(self, config: Config, batch_data_items: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert config is not None
        assert batch_data_items is not None

        self.config = config
        self.current_batch_episode_idx = batch_data_items["episode_idx"]
        self.batch_agent_start_pos = batch_data_items["agent_start_pos"]
        self.batch_agent_target_pos = batch_data_items["agent_target_pos"]
        self.batch_agent_current_pos = batch_data_items["agent_current_pos"]
        self.fov = batch_data_items["fov"].clone()
        self.fov_block_mask = batch_data_items["fov_block_mask"].clone()
        self.batch_action_idx_history = None
        self.batch_logit_prob_history = None
        self.batch_top_k_prob_history = None
        self.batch_at_target_position_mask = None

    def update_step(
        self,
        batch_action_idx: torch.Tensor,
        batch_logit_prob: torch.Tensor,
        batch_top_k_prob: torch.Tensor,
        step: int,
        debug: bool,
    ):
        """
        Update the record with the current step context info.
        """
        # assert 0 <= step and step < self.config.episode_steps, f"invalid step: {step}"

        # Get the action update
        batch_actions: torch.Tensor = self.config.possible_actions[
            batch_action_idx.squeeze(dim=1)
        ]
        batch_agent_next_pos = self.batch_agent_current_pos + batch_actions
        batch_agent_next_pos[:, 0] = torch.clamp(
            batch_agent_next_pos[:, 0],
            min=self.config.world_min_x,
            max=self.config.world_max_x - 1,
        )

        batch_agent_next_pos[:, 1] = torch.clamp(
            batch_agent_next_pos[:, 1],
            min=self.config.world_min_y,
            max=self.config.world_max_y - 1,
        )

        B, _ = batch_agent_next_pos.size()
        x_indices = batch_agent_next_pos[:, 0].to(torch.int)
        y_indices = batch_agent_next_pos[:, 1].to(torch.int)
        blocked_pos_mask = (
            self.fov[torch.arange(B), y_indices, x_indices]
            == self.config.ENCODE_BLOCK  # Row - Y-axis, Col - X-axis
        )
        reach_target_pos_mask = (
            self.fov[torch.arange(B), y_indices, x_indices]
            == self.config.ENCODE_TARGET_POS
        )
        if self.batch_at_target_position_mask is None:
            self.batch_at_target_position_mask = reach_target_pos_mask
        else:
            self.batch_at_target_position_mask |= reach_target_pos_mask

        # Action overwrite:
        #  - cannot move onto the BLOCK
        #  - if already at the TARGET position, no more move
        batch_actions[blocked_pos_mask] = torch.tensor(
            [0, 0], device=self.config.device
        )

        # batch_actions[self.batch_at_target_position_mask] = torch.tensor([0, 0])
        self.batch_agent_current_pos += batch_actions
        self.batch_agent_current_pos[:, 0] = torch.clamp(
            self.batch_agent_current_pos[:, 0],
            min=self.config.world_min_x,
            max=self.config.world_max_x - 1,
        )

        self.batch_agent_current_pos[:, 1] = torch.clamp(
            self.batch_agent_current_pos[:, 1],
            min=self.config.world_min_y,
            max=self.config.world_max_y - 1,
        )

        # Update the fov
        x_indices = self.batch_agent_current_pos[:, 0].to(torch.int)
        y_indices = self.batch_agent_current_pos[:, 1].to(torch.int)
        self.fov = self.fov.clone()
        self.fov[torch.arange(B), y_indices, x_indices] = (
            self.config.ENCODE_START_STEP_IDX + step  # Row - Y-axis, Col - X-axis
        )

        # Save the history -- batch_action_idx
        if self.batch_action_idx_history is None:
            self.batch_action_idx_history = batch_action_idx
        else:
            self.batch_action_idx_history = torch.vstack(
                (self.batch_action_idx_history, batch_action_idx)
            )

        # Save the history -- batch_logit_prob
        if self.batch_logit_prob_history is None:
            self.batch_logit_prob_history = batch_logit_prob
        else:
            self.batch_logit_prob_history = torch.vstack(
                (self.batch_logit_prob_history, batch_logit_prob)
            )

        # Save the history -- batch_top_k_prob
        if self.batch_top_k_prob_history is None:
            self.batch_top_k_prob_history = batch_top_k_prob
        else:
            self.batch_top_k_prob_history = torch.vstack(
                (self.batch_top_k_prob_history, batch_top_k_prob)
            )

        assert (
            self.batch_action_idx_history.size()
            == self.batch_logit_prob_history.size()
            == self.batch_top_k_prob_history.size()
        )

    def viz_fov(
        self,
        ax: matplotlib.axes._axes.Axes,
        idx: int = 0,
        reward_model: RewardModel = None,
    ):
        """
        Visualize the:
         - fov
         - the state transition history
         - the last state position
        """
        assert ax is not None
        ax.pcolormesh(self.fov[idx], cmap=cm.gray, edgecolors="gray", linewidths=0.5)
        if reward_model is not None:
            rewards = self.reward(reward_model=reward_model)

            # B x G x S
            batch_logit_prob_history = self.batch_logit_prob_history.reshape(
                (-1, self.config.episode_steps)
            )
            ax.set_title(
                f"episode: {self.current_batch_episode_idx[idx]}: reward: {rewards[idx].item():.2f}"
            )

            cur_pos = self.batch_agent_current_pos[idx]
            x0 = int(cur_pos[0])
            y0 = int(cur_pos[1])
            ax.annotate(
                f"final:p{batch_logit_prob_history[idx][-1].item()*100.:.1f}%",
                xy=(x0, y0),
                xycoords="data",
                color="red",
                fontsize=12,
            )

    def reward(self, reward_model: RewardModel) -> torch.tensor:
        """
        Use the given reward model to compute the final state reward.
        """
        assert reward_model is not None
        return reward_model.reward(
            batach_cur_pos=self.batch_agent_current_pos,
            batch_target_pos=self.batch_agent_target_pos,
        )
