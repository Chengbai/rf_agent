from __future__ import annotations

from dataclasses import dataclass

from src.config import Config


@dataclass
class Action:
    possible_actions: list
    action_idx: int | None

    @staticmethod
    def create_from_config(config: Config) -> Action:
        return Action(possible_actions=config.possible_actions)

    def get_udpate(self):
        return self.possible_actions[self.action_idx]

    def take_action(self, action_idx: int):
        assert 0 <= action_idx and action_idx < len(self.possible_actions)
        self.action_idx = action_idx
        return self.get_udpate()
