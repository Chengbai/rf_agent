# Created 2025
import torch.nn as nn

from src.rl_data_record import RLDataRecord


class PolicyBaseModel(nn.Module):
    """Base Policy model"""

    def __init__(self, *args, **kwargs):
        """Init the class instance."""
        super().__init__(*args, **kwargs)

    def forward(self, batch_rl_data_record: RLDataRecord) -> None:
        """forward placeholder."""
        assert batch_rl_data_record is not None

        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "forward" function'
        )
