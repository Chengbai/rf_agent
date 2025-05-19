import pytest
import torch
from dataclasses import FrozenInstanceError

from config import Config


def test_config_creation():
    config = Config()
    assert isinstance(config, Config)


def test_config_world_dimensions():
    config = Config()
    assert config.world_width == 31
    assert config.world_height == 31


def test_config_possible_actions():
    config = Config()
    assert isinstance(config.possible_actions, torch.Tensor)
    assert config.possible_actions.shape == (9, 2)


def test_config_fov_dimensions():
    config = Config()
    assert config.field_of_view_width == 15
    assert config.field_of_view_height == 15


def test_config_device():
    config = Config()
    assert config.device == torch.device("cpu")


def test_possible_actions():
    config = Config()
    batch_action_idices = torch.arange(len(config.possible_actions))
    batch_actions = config.possible_actions[batch_action_idices]
    print(batch_actions.size(), batch_actions)
    assert torch.equal(batch_actions, config.possible_actions)
