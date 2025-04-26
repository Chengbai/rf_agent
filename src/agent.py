from __future__ import annotations

from dataclasses import dataclass

from src.state import State


@dataclass
class Agent:
    id: str
    state: State

    @staticmethod
    def create_from(id: str, state: State) -> Agent:
        assert id
        assert state is not None
        return Agent(id=id, state=state)
