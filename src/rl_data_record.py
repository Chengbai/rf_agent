import torch
import torch.nn as nn

from src.config import Config


class RLDataRecord(nn.Module):
    def __init__(self, config: Config, batch_data_items: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert config is not None
        assert batch_data_items is not None
        self.config = config
        self.NO_MOVE_ACTION_INDEX = config.possible_actions.shape[0] // 2

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
        debug: bool = False,
    ):
        # assert 0 <= step and step < self.config.episode_steps, f"invalid step: {step}"

        # Get the action update
        batch_actions: torch.Tensor = self.config.possible_actions[
            batch_action_idx.squeeze(dim=1)
        ]
        batch_agent_next_pos = self.batch_agent_current_pos + batch_actions
        batch_agent_next_pos[:, 0] = torch.clamp(
            batch_agent_next_pos[:, 0],
            min=self.config.world_min_y,
            max=self.config.world_max_y,
        )

        batch_agent_next_pos[:, 1] = torch.clamp(
            batch_agent_next_pos[:, 1],
            min=self.config.world_min_x,
            max=self.config.world_max_x,
        )

        B, positions = batch_agent_next_pos.size()
        y_indices = batch_agent_next_pos[:, 0]
        x_indices = batch_agent_next_pos[:, 1]
        blocked_pos_mask = (
            self.fov[torch.arange(B), y_indices, x_indices] == self.config.ENCODE_BLOCK
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

        # Action Index Overwrite
        # batch_action_idx[self.batch_at_target_position_mask] = self.NO_MOVE_ACTION_INDEX

        # Update the fov
        y_indices = self.batch_agent_current_pos[:, 0]
        x_indices = self.batch_agent_current_pos[:, 1]
        self.fov = self.fov.clone()
        self.fov[torch.arange(B), y_indices, x_indices] = (
            self.config.ENCODE_START_STEP_IDX + step
        )

        # Save the history -- batch_action_idx
        if self.batch_action_idx_history is None:
            self.batch_action_idx_history = batch_action_idx
        else:
            self.batch_action_idx_history = torch.vstack(
                (self.self.batch_action_idx_history, batch_action_idx)
            )

        # Save the history -- batch_logit_prob
        if self.batch_logit_prob_history is None:
            self.batch_logit_prob_history = batch_logit_prob
        else:
            self.batch_logit_prob_history = torch.vstack(
                (self.self.batch_logit_prob_history, batch_logit_prob)
            )

        # Save the history -- batch_top_k_prob
        if self.batch_top_k_prob_history is None:
            self.batch_top_k_prob_history = batch_top_k_prob
        else:
            self.batch_top_k_prob_history = torch.vstack(
                (self.self.batch_top_k_prob_history, batch_top_k_prob)
            )

        assert (
            self.batch_action_idx_history.size()
            == self.batch_logit_prob_history.size()
            == self.batch_top_k_prob_history.size()
        )
