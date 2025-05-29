# Created on 2025
from enum import Enum


from src.config import Config
from src.policy.linear_model_policy import LinearModelPolicy
from src.policy.linear_model_with_later_position_fusion_policy import (
    LinearModelWithLaterPositionFusionPolicy,
)
from src.policy.transformer_model_policy import TransformerPolicy


class PolicyMode(Enum):
    """An enum defines the supported policy model modes."""

    LINEAR_MODEL = 1
    LINEAR_MODEL_WITH_LATE_POSITION_FUSION = 2
    TRANSFORMER_WITH_LATE_POSITION_FUSION = 3


class PolicyFactory:
    """Policy factory to create policy model based on the given mode and config."""

    @staticmethod
    def create(
        policy_mode: PolicyMode,
        config: Config,
    ):
        """Create and return a new Policy model instance based on the given mode and config."""
        if policy_mode == PolicyMode.LINEAR_MODEL:
            return LinearModelPolicy(config=config).to(config.device)
        elif policy_mode == PolicyMode.LINEAR_MODEL_WITH_LATE_POSITION_FUSION:
            return LinearModelWithLaterPositionFusionPolicy(config=config).to(
                config.device
            )
        elif policy_mode == PolicyMode.TRANSFORMER_WITH_LATE_POSITION_FUSION:
            return TransformerPolicy(config=config).to(config.device)
        else:
            raise NotImplementedError(f"Not supported policy mode: {policy_mode}")
