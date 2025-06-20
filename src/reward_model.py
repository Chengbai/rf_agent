import torch

from src.action import Action
from src.agent import Agent
from src.config import Config
from src.state import State
from src.world import World


class RewardModel:
    def __init__(self, config: Config):
        assert config is not None
        self.config = config

    def actionreward(
        self, world: World, agent: Agent, state: State, action: Action
    ) -> float:
        assert world is not None
        assert agent is not None
        assert state is not None
        assert action is not None

        dxy = torch.tensor(action.get_udpate())
        cur_pos: torch.Tensor = state.position()  # [x, y]
        pre_pos = cur_pos - dxy

        target_position = agent.target_state.position()
        if torch.equal(cur_pos, target_position):
            # Reach the target. Big reward!
            return self.config.max_reward

        # REVIEW: this is a post action learning logic.
        # If model is very hard to learn, we need rethink here
        if not world.can_move_to(cur_pos):
            return self.config.blocked_reward

        # moving closer, get reward
        # l2-norm
        return torch.linalg.vector_norm(
            pre_pos - target_position, ord=2
        ) - torch.linalg.vector_norm(cur_pos - target_position, ord=2)

    def state_reward(self, world: World, agent: Agent, state: State) -> float:
        assert world is not None
        assert agent is not None
        assert state is not None

        cur_pos: torch.Tensor = state.position()  # [x, y]
        target_position = agent.target_state.position()
        if torch.equal(cur_pos, target_position):
            # Reach the target. Big reward!
            return self.config.max_reward

        # REVIEW: this is a post action learning logic.
        # If model is very hard to learn, we need rethink here
        if not world.can_move_to(cur_pos):
            return self.config.blocked_reward

        # moving closer, get reward
        # l2-norm
        return -torch.linalg.vector_norm(cur_pos - target_position, ord=2)

    def reward(
        self, batach_cur_pos: torch.Tensor, batch_target_pos: torch.Tensor
    ) -> float:
        assert batach_cur_pos is not None
        assert batch_target_pos is not None

        # moving closer, get reward
        # l2-norm
        batch_rewards = -torch.linalg.vector_norm(
            batach_cur_pos - batch_target_pos, dim=1, ord=2
        )
        batch_rewards[batch_rewards == 0.0] = (
            10.0  # big reward if end at the target position
        )
        return batch_rewards
