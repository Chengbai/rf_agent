from __future__ import annotations

import matplotlib
import random
import torch
import torch.nn.functional as F

from dataclasses import dataclass

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
    id: str
    config: Config
    world: World
    agent: Agent

    @staticmethod
    def new(id: str) -> Episode:
        config = Config()
        world = World.create_from_config(id=id, config=config)

        id = id if id else "earth"
        start_state = State.create_from(
            config=config,
            id=id,
            x=(config.world_min_x + config.world_max_x) // 2,
            y=(config.world_min_y + config.world_max_y) // 2,
        )
        target_state = State.create_from(
            config=config,
            id=id if id else "earth",
            x=random.uniform(config.world_min_x, config.world_max_x),
            y=random.uniform(config.world_min_y, config.world_max_y),
        )
        agent = Agent.create_from(
            id=id, start_state=start_state, target_state=target_state
        )
        return Episode(id=id, config=config, world=world, agent=agent)

    @staticmethod
    def create_from_state(
        start_state: State, target_state: State, id: str = None
    ) -> Episode:
        assert start_state is not None
        assert target_state is not None
        assert start_state.config is not None
        assert start_state.config == target_state.config

        world = World.create_from_config(
            id=id if id else "world", config=start_state.config
        )
        agent = Agent.create_from(
            id=id if id else "earth", start_state=start_state, target_state=target_state
        )
        return Episode(id=id, config=start_state.config, world=world, agent=agent)

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
            agent_current_pos = self.agent.current_state.normalized_position()[
                None, :
            ]  # 1 x 2
            agent_target_pos = self.agent.target_state.normalized_position()[
                None, :
            ]  # 1 x 2

            fov = self.world.fov(center_pos=self.agent.current_state.position())
            batch_fov = fov.reshape(shape=(-1,))[None, :]
            batch_features = torch.concat(
                [
                    agent_current_pos,
                    agent_target_pos,
                    batch_fov,
                ],
                dim=1,
            )

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
                    f"step: {step}, logits: {batch_logits}, logit_prob: {batch_logit_prob}, top_k_prob: {batch_top_k_prob}, action_idx: {batch_action_idx}, state: {self.agent.current_state.normalized_position()}"
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
        return torch.sum(
            torch.tensor(
                [
                    reward_model.reward(
                        world=self.world, agent=self.agent, state=state, action=action
                    )
                    for state, action in zip(
                        self.agent.state_history, self.agent.action_history
                    )
                ]
            )
        )

    def log_reward_prob(self) -> torch.tensor:
        return self.agent.log_reward_prob()

    def reward_prob(self) -> torch.tensor:
        return self.agent.reward_prob()

    def viz(self, ax: matplotlib.axes._axes.Axes, color: str = "red"):
        self.world.viz(ax=ax, agent=self.agent, config=self.config, color=color)

    def take_action(self, action: Action):
        return self.agent.take_action(action)

    def reset(self):
        self.agent.reset()
        self.world.reset()
