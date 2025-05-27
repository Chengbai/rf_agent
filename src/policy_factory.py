# Created on 2025
from enum import Enum


from src.config import Config
from src.policy.linear_model_policy import LinearModelPolicy
from src.policy.linear_model_with_later_position_fusion_policy import (
    LinearModelWithLaterPositionFusionPolicy,
)


class PolicyMode(Enum):
    """An enum defines the supported policy model modes."""

    LINEAR_MODEL = 1
    LINEAR_MODEL_WITH_LATE_POSITION_FUSION = 2


class PolicyFactory:
    """Policy factory to create policy model based on the given mode and config."""

    @staticmethod
    def create(
        policy_mode: PolicyMode,
        config: Config,
    ):
        """Create and return a new Policy model instance based on the given mode and config."""
        if policy_mode == PolicyMode.LINEAR_MODEL:
            return LinearModelPolicy(config=config)
        elif policy_mode == PolicyMode.LINEAR_MODEL_WITH_LATE_POSITION_FUSION:
            return LinearModelWithLaterPositionFusionPolicy(config=config)
        else:
            raise NotImplementedError(f"Not supported policy mode: {policy_mode}")
