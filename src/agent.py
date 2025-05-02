from __future__ import annotations

import copy
import torch

from dataclasses import dataclass

from src.action import Action
from src.state import State


@dataclass
class Agent:
    id: str
    init_state: State
    state: State
    action_history: list[Action] | None

    @staticmethod
    def create_from(id: str, state: State) -> Agent:
        assert id
        assert state is not None
        return Agent(
            id=id, init_state=copy.deepcopy(state), state=state, action_history=[]
        )

    def take_action(self, action: Action) -> Agent:
        assert action is not None

        self.state.take_action(action=action)
        self.action_history.append(action)
        return self

    def reward(self) -> torch.tensor:
        return sum([action.reward() for action in self.action_history])

    def log_reward_prob(self) -> torch.tensor:
        # The chain of the action probability (CoA)
        # P(CoA) = a1.prob * a2.prob * ...
        # Log(P(CoA)) = log(a1.prob) + log(a2.prob) + ...
        # NOTE: use 'sum' to track the gradient
        return sum([torch.log(action.prob) for action in self.action_history])

    def reward_prob(self) -> torch.tensor:
        # The chain of the action probability (CoA)
        # P(CoA) = a1.prob * a2.prob * ...
        prob = 1.0
        for action in self.action_history:
            prob *= action.prob

        return prob
