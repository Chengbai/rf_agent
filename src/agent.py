from __future__ import annotations

import copy
import torch

from dataclasses import dataclass

from src.action import Action
from src.state import State


@dataclass
class Agent:
    id: str
    start_state: State
    target_state: State
    current_state: State
    state_history: list[State] | None
    action_history: list[Action] | None

    @staticmethod
    def create_from(id: str, start_state: State, target_state: State) -> Agent:
        assert id
        assert start_state is not None
        assert target_state is not None
        return Agent(
            id=id,
            start_state=copy.deepcopy(start_state),
            target_state=copy.deepcopy(target_state),
            current_state=copy.deepcopy(start_state),
            state_history=[],
            action_history=[],
        )._init()

    def _init(self) -> Agent:
        self.state_history = [self.start_state]
        return self

    def take_action(self, action: Action) -> Agent:
        assert action is not None

        # check is at the `target_state`, if it is no more move
        if torch.equal(self.current_state.position(), self.target_state.position()):
            return self

        # check is out of the border. If it is no more move
        if self.current_state.can_take_action(action=action):
            self.current_state.take_action(action=action)

            self.action_history.append(action)

            # `state_history` contains all of state afte the `start_state`
            self.state_history.append(self.current_state.copy())
        return self

    def reward(self) -> torch.tensor:
        return sum([action.reward() for action in self.action_history])

    def log_reward_prob(self) -> torch.tensor:
        # The chain of the action probability (CoA)
        # P(CoA) = a1.prob * a2.prob * ...
        # Log(P(CoA)) = log(a1.prob) + log(a2.prob) + ...
        # NOTE: use 'sum' to track the gradient\
        if not self.action_history:
            return torch.tensor([0.0])
        return sum([torch.log(action.prob) for action in self.action_history])

    def reward_prob(self) -> torch.tensor:
        # The chain of the action probability (CoA)
        # P(CoA) = a1.prob * a2.prob * ...
        if not self.action_history:
            return torch.tensor([0.0])
        prob = 1.0
        for action in self.action_history:
            prob *= action.prob

        return prob

    def reset(self):
        self.current_state = copy.deepcopy(self.start_state)
        self.action_history = []
