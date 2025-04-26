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


@dataclass
class Episode:
    config: Config
    world: World
    agent: Agent
    policy: Policy

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
        policy = Policy(config=config)

        return Episode(config=config, world=world, agent=agent, policy=policy)

    def run_steps_by_random(self, steps: int, debug: bool = False):
        for step in range(steps):
            action_idx = random.randint(0, len(self.config.possible_actions) - 1)
            self.agent.take_action(
                action=Action.create_from(config=self.config, action_idx=action_idx)
            )

    def run_steps_by_policy(self, steps: int, debug: bool = False):
        for step in range(steps):
            pred = self.policy.forward(self.agent.state.to_tensor())
            action_idx = torch.argmax(pred)
            self.agent.take_action(
                action=Action.create_from(config=self.config, action_idx=action_idx)
            )
            if debug:
                print(
                    f"step: {step}, pred: {pred.detach()}, sum_pred: {torch.sum(pred)}, action_idx: {action_idx}, state: {self.agent.state.to_tensor()}"
                )

    def viz(self, ax: matplotlib.axes._axes.Axes):
        self.world.viz(ax=ax, agent=self.agent, config=self.config)
