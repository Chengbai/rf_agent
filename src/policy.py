import torch
import torch.nn as nn

from src.config import Config


class Policy(nn.Module):
    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert config is not None
        self._config = config

        self.brain = nn.Sequential(
            nn.Linear(config.input_features, config.intermedia_features, bias=True),
            nn.ReLU(),
            nn.Linear(config.intermedia_features, len(config.possible_actions)),
            nn.Softmax(),
        )

    def forward(self, x: torch.tensor):
        assert x is not None
        assert x.shape == (2,)
        return self.brain(x)
