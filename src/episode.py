from __future__ import annotations

import matplotlib
import random

from dataclasses import dataclass

import torch


from src.action import Action
from src.agent import Agent
from src.config import Config
from src.policy import Policy
from src.reward_model import RewardModel
from src.state import State
from src.world import World
import src.utils as utils


@dataclass
class Episode:
    """The Episode object in the RL."""

    episode_id: str
    config: Config
    world: World
    agent: Agent

    @staticmethod
    def new(episode_id: str) -> Episode:
        config = Config()
        world = World.create_from_config(world_id=episode_id, config=config)

        episode_id = episode_id if episode_id else "start_state"
        start_state = State.create_from(
            config=config,
            state_id=episode_id,
            # x=0,
            # y=0,
            x=(config.world_min_x + config.world_max_x) // 2,
            y=(config.world_min_y + config.world_max_y) // 2,
        )
        target_state = State.create_from(
            config=config,
            state_id=episode_id if episode_id else "target_state",
            # x=0.0,
            # y=0.0,
            x=float(config.world_min_x),
            y=config.world_max_y - 1.0,
            # x=math.floor(random.uniform(config.world_min_x, config.world_max_x - 1.0))
            # * 1.0,
            # y=math.floor(random.uniform(config.world_min_y, config.world_max_y - 1.0))
            # * 1.0,
        )
        agent = Agent.create_from(
            agent_id=episode_id, start_state=start_state, target_state=target_state
        )
        return Episode(episode_id=episode_id, config=config, world=world, agent=agent)

    @staticmethod
    def create_from_state(
        start_state: State, target_state: State, episode_id: str = None
    ) -> Episode:
        assert start_state is not None
        assert target_state is not None
        assert start_state.config is not None
        assert start_state.config == target_state.config

        world = World.create_from_config(
            id=episode_id if episode_id else "world", config=start_state.config
        )
        agent = Agent.create_from(
            agent_id=episode_id if episode_id else "earth",
            start_state=start_state,
            target_state=target_state,
        )
        return Episode(
            episode_id=episode_id, config=start_state.config, world=world, agent=agent
        )

    def clone(self, repeat: int) -> list[Episode]:
        assert repeat > 0

        clone_episodes = []
        for i in range(repeat):
            clone_episodes.append(
                Episode(
                    episode_id=f"{self.episode_id}:clone:{i}",
                    config=self.config,
                    world=self.world.clone(idx=i),
                    agent=self.agent.clone(idx=i),
                )
            )
        return clone_episodes

    def run_steps_by_random(self, steps: int, debug: bool = False):
        for step in range(steps):
            action_idx = random.randint(0, len(self.config.possible_actions) - 1)
            self.agent.take_action(
                action=Action.create_from(
                    config=self.config, action_idx=action_idx, prob=torch.tensor(1.0)
                ),
            )

    def _run_steps(self, steps: int, policy: Policy, top_k: int, debug: bool = False):
        for step in range(steps):
            # logits = policy.forward(self.agent.current_state.normalized_position())
            agent_current_pos = self.agent.current_state.position()[None, :]  # 1 x 2
            agent_target_pos = self.agent.target_state.position()[None, :]  # 1 x 2

            fov = self.fov(center_pos=self.agent.current_state.position())
            batch_fov = fov.reshape(shape=(-1,))[None, :]
            batch_features = batch_fov
            batch_features = batch_features.to(next(policy.parameters()).device)

            batch_logits = policy.forward(batch_features)
            batch_action_idx, batch_logit_prob, batch_top_k_prob = utils.top_k_sampling(
                logits=batch_logits, k=top_k
            )
            # action_idx = torch.argmax(logits)

            self.agent.take_action(
                action=Action.create_from(
                    config=self.config,
                    action_idx=batch_action_idx[0][0],
                    prob=batch_logit_prob[0],
                )
            )
            if debug:
                print(
                    f"step: {step}, logits: {batch_logits}, logit_prob: {batch_logit_prob}, top_k_prob: {batch_top_k_prob}, action_idx: {batch_action_idx}, state: {self.agent.current_state.position()}, action_history: {[a.action_idx for a in self.agent.action_history]}"
                )

    def run_steps_by_policy(
        self, steps: int, policy: Policy, top_k: int, debug: bool = False
    ):
        with torch.no_grad():
            policy.eval()
            self._run_steps(steps=steps, policy=policy, top_k=top_k, debug=debug)

    def inference_steps_by_policy(
        self, steps: int, policy: Policy, debug: bool = False
    ):
        self.run_steps_by_policy(steps=steps, policy=policy, top_k=1, debug=debug)

    def train(self, steps: int, policy: Policy, debug: bool = False):
        policy.train()
        self._run_steps(
            steps=steps, policy=policy, top_k=self.config.top_k, debug=debug
        )

    def reward(self, reward_model: RewardModel) -> torch.tensor:
        assert reward_model is not None
        # return reward_model.state_reward(
        #     world=self.world,
        #     agent=self.agent,
        #     state=self.agent.state_history[-1],
        # )
        return reward_model.reward(
            batach_cur_pos=self.agent.current_state.position()[None, :],
            batch_target_pos=self.agent.target_state.position()[None, :],
        )

    def log_reward_prob(self) -> torch.tensor:
        return self.agent.log_reward_prob()

    def reward_prob(self) -> torch.tensor:
        return self.agent.reward_prob()

    def fov(self, center_pos: torch.tensor) -> torch.tensor:
        world_fov = self.world.fov(center_pos=center_pos)

        # Encode the fov
        cy = int(center_pos[0])  # Rows -> y-axis
        cx = int(center_pos[1])  # Columns -> x-axis
        world_fov[cx, cy] = self.config.ENCODE_START_POS

        target_pos = self.agent.target_state.position()
        ty = int(target_pos[0])  # Rows -> y-axis
        tx = int(target_pos[1])  # Columns -> x-axis
        world_fov[tx, ty] = self.config.ENCODE_TARGET_POS

        for state_idx, state in enumerate(self.agent.state_history):
            state_pos = state.position()
            sy = int(state_pos[0])  # Rows -> y-axis
            sx = int(state_pos[1])  # Columns -> x-axis
            world_fov[sx, sy] = self.config.ENCODE_START_STEP_IDX

        return world_fov

    def viz(
        self,
        ax: matplotlib.axes._axes.Axes,
        reward_model: RewardModel,
        color: str = "red",
    ):
        self.world.viz(
            ax=ax,
            agent=self.agent,
            config=self.config,
            color=color,
        )
        rewards = self.reward(reward_model=reward_model)
        ax.set_title(f"{self.episode_id}: {rewards}")

    def viz_fov(self, ax: matplotlib.axes._axes.Axes):
        assert ax is not None
        fov = self.fov(center_pos=self.agent.start_state.position())
        ax.pcolormesh(fov, cmap=self.config.CMAP, edgecolors="gray", linewidths=0.5)

    def take_action(self, action: Action):
        return self.agent.take_action(action)

    def reset(self):
        self.agent.reset()
        self.world.reset()
