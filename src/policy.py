import torch
import torch.nn as nn

from src.config import Config


class Policy(nn.Module):
    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert config is not None
        self._config = config

        self.brain = nn.Sequential(
            nn.Linear(config.input_features, config.intermedia_features1, bias=True),
            nn.ReLU(),
            nn.Linear(
                config.intermedia_features1, config.intermedia_features2, bias=True
            ),
            nn.ReLU(),
            nn.Linear(config.intermedia_features2, len(config.possible_actions)),
        )
        self._init_parameters()

    def _init_parameters(self):
        for layer in self.brain:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x: torch.tensor):
        assert x is not None
        return self.brain(x)
