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
from src.state import State
from src.world import World
import src.utils as utils


@dataclass
class Episode:
    config: Config
    world: World
    agent: Agent

    @staticmethod
    def new() -> Episode:
        config = Config()
        world = World.create_from_config(config=config)

        state = State.create_from(
            config=config,
            id="earth",
            x=random.uniform(config.world_min_x, config.world_max_x),
            y=random.uniform(config.world_min_y, config.world_max_y),
        )
        agent = Agent.create_from(id="earth", state=state)
        return Episode(config=config, world=world, agent=agent)

    @staticmethod
    def create_from_state(state: State) -> Episode:
        assert state is not None
        world = World.create_from_config(config=state.config)
        agent = Agent.create_from(id="earth", state=state)
        return Episode(config=state.config, world=world, agent=agent)

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
            logits = policy.forward(self.agent.state.to_tensor())
            action_idx, logit_prob, top_k_prob = utils.top_k_sampling(
                logits=logits, k=top_k
            )
            # action_idx = torch.argmax(logits)

            self.agent.take_action(
                action=Action.create_from(
                    config=self.config, action_idx=action_idx, prob=logit_prob
                )
            )
            if debug:
                print(
                    f"step: {step}, logits: {logits}, logit_prob: {logit_prob}, top_k_prob: {top_k_prob}, action_idx: {action_idx}, state: {self.agent.state.to_tensor()}"
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

    def reward(self) -> torch.tensor:
        return self.agent.reward()

    def log_reward_prob(self) -> torch.tensor:
        return self.agent.log_reward_prob()

    def reward_prob(self) -> torch.tensor:
        return self.agent.reward_prob()

    def viz(self, ax: matplotlib.axes._axes.Axes, color: str = "red"):
        self.world.viz(ax=ax, agent=self.agent, config=self.config, color=color)
