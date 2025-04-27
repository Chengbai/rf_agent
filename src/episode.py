from __future__ import annotations

import matplotlib
import random
import torch

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

    def run_steps_by_random(self, steps: int, debug: bool = False):
        for step in range(steps):
            action_idx = random.randint(0, len(self.config.possible_actions) - 1)
            self.agent.take_action(
                action=Action.create_from(
                    config=self.config, action_idx=action_idx, prob=torch.tensor(1.0)
                ),
            )

    def run_steps_by_policy(self, steps: int, policy: Policy, debug: bool = False):
        with torch.no_grad():
            policy.eval()
            for step in range(steps):
                logits = policy.forward(self.agent.state.to_tensor())
                action_idx, logit_prob, top_k_prob = utils.top_k_sampling(
                    logits=logits, k=self.config.top_k
                )
                # action_idx = torch.argmax(logits)

                self.agent.take_action(
                    action=Action.create_from(
                        config=self.config, action_idx=action_idx, prob=logit_prob
                    )
                )
                if debug:
                    print(
                        f"step: {step}, logit_prob: {logit_prob}, top_k_prob: {top_k_prob}, action_idx: {action_idx}, state: {self.agent.state.to_tensor()}"
                    )

    def gain(self):
        return self.agent.gain()

    def train(self, steps: int, policy: Policy, debug: bool = False):
        policy.train()

    def viz(self, ax: matplotlib.axes._axes.Axes):
        self.world.viz(ax=ax, agent=self.agent, config=self.config)
