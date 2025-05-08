import torch

from dataclasses import dataclass, field


@dataclass
class Config:
    # world
    world_min_x: int = 0
    world_max_x: int = 30
    world_min_y: int = 0
    world_max_y: int = 20
    world_width: int = int(world_max_x - world_min_x)
    world_height: int = int(world_max_y - world_min_y)
    world_block_probability = 0.2

    # Dataset
    train_dataset_length: int = 10000
    train_batch_size: int = 50

    test_dataset_length: int = 2000
    test_batch_size: int = 20

    # Train
    epoches: int = 2

    # Actions
    possible_actions: torch.tensor = field(
        default_factory=lambda: torch.tensor(
            [(1 * i, 1 * j) for i in range(-1, 2) for j in range(-1, 2)]
        )
    )

    # Policy
    field_of_view: int = 5  # FOV

    # input feature: [B, current_pos, target_pos, fov]
    input_features: int = 2 + 2 + (2 * field_of_view + 1) ** 2

    intermedia_features1: int = 500
    intermedia_features2: int = 300
    intermedia_features3: int = 100

    # output feature: [B, 1] -> action_idx
    output_features: int = 1

    # Policy sampling
    top_k: int = 5

    # GRPO policy training
    episode_steps: int = 10
    # episodes_per_iteration: int = 2

    # EPSILON
    epsilon = 1e-5

    # Device
    device = torch.device("mps")

    # Figure size
    figure_size: tuple[int, int] = (5, 5)

    # Rewards
    max_reward = torch.tensor(10.0)
    blocked_reward = torch.tensor(-5.0)
