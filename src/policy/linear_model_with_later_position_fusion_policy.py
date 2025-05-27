import torch
import torch.nn as nn

from src.config import Config
from src.policy.policy_base import PolicyBaseModel
from src.rl_data_record import RLDataRecord


class LinearModelWithLaterPositionFusionPolicy(PolicyBaseModel):
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
            nn.Linear(
                config.intermedia_features2, config.intermedia_features3, bias=True
            ),
            nn.ReLU(),
        )
        self.mlp = (
            nn.Linear(config.intermedia_features3 + 4, len(config.possible_actions)),
        )

        self._init_parameters()

    def _init_parameters(self):
        for layer in self.brain:
            if isinstance(layer, nn.Linear):
                layer.weight = nn.init.kaiming_uniform(layer.weight)

    def _prepare_feature(self, batch_rl_data_record: RLDataRecord) -> torch.Tensor:
        """Prepare the train/eval feature data for the model"""

        assert batch_rl_data_record is not None
        B, _, _ = batch_rl_data_record.fov.shape
        batch_fov = batch_rl_data_record.fov.reshape(shape=(B, -1))
        return batch_fov

    def forward(self, batch_rl_data_record: RLDataRecord):
        batch_features = self._prepare_feature(
            batch_rl_data_record=batch_rl_data_record
        )
        return self.brain(batch_features)
        assert batch_x is not None
        assert batch_cur_position is not None
        assert batch_target_position is not None
        assert (
            batch_x.size(0)
            == batch_cur_position.size(0)
            == batch_target_position.size(0)
        )

        batch_feature = self.brain(batch_x)

        batch_feature = torch.concat(
            [batch_feature, batch_cur_position, batch_target_position]
        )
        logits = self.mlp(batch_feature)
        return logits
