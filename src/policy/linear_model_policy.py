import torch
import torch.nn as nn

from src.config import Config
from src.policy.policy_base import PolicyBaseModel
from src.rl_data_record import RLDataRecord


class LinearModelPolicy(PolicyBaseModel):
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
                layer.weight = nn.init.kaiming_uniform(layer.weight)

    def prepare_feature(self, batch_rl_data_record: RLDataRecord) -> torch.Tensor:
        """Prepare the train/eval feature data for the model"""

        assert batch_rl_data_record is not None
        B, _, _ = batch_rl_data_record.fov.shape
        batch_fov = batch_rl_data_record.fov.reshape(shape=(B, -1))
        return batch_fov

    def forward(self, batch_fov: torch.Tensor):
        assert batch_fov is not None
        return self.brain(batch_fov)

    def execute_1_step(self, batch_rl_data_record: RLDataRecord) -> torch.Tensor:
        batch_fov = self.prepare_feature(batch_rl_data_record=batch_rl_data_record)
        assert batch_fov is not None
        return self.forward(
            batch_fov=batch_fov,
        )
