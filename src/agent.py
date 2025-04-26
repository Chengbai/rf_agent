from __future__ import annotations

from dataclasses import dataclass

from src.action import Action
from src.state import State


@dataclass
class Agent:
    id: str
    state: State
    action_history: list[Action] | None

    @staticmethod
    def create_from(id: str, state: State) -> Agent:
        assert id
        assert state is not None
        return Agent(id=id, state=state, action_history=[])

    def take_action(self, action: Action) -> Agent:
        assert action is not None

        self.state.take_action(action=action)
        self.action_history.append(action)
        return self
