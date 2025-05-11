import torch

from dataclasses import dataclass, field


@dataclass
class Config:
    # world
    world_min_x: int = 0
    world_max_x: int = 51
    world_min_y: int = 0
    world_max_y: int = 51
    world_width: int = int(world_max_x - world_min_x)
    world_height: int = int(world_max_y - world_min_y)
    world_block_probability = 0.2

    # Dataset
    train_dataset_length: int = 1000
    train_batch_size: int = 20

    test_dataset_length: int = 10
    test_batch_size: int = 5

    eval_dataset_length: int = 100
    eval_batch_size: int = 5

    # Actions
    possible_actions: torch.tensor = field(
        default_factory=lambda: torch.tensor(
            [(1 * i, 1 * j) for i in range(-1, 2) for j in range(-1, 2)]
        )
    )

    # FOV
    field_of_view_width: int = world_width // 2  # FOV width
    field_of_view_height: int = world_height // 2  # FOV height

    ENCODE_BLOCK: int = 0
    ENCODE_START_POS: int = 128
    ENCODE_START_STEP_IDX = 150
    ENCODE_TARGET_POS: int = 200
    ENCODE_EMPTY: int = 255

    ENCODE_COLORS = ["black", "red", "blue", "yellow", "white"]

    # Policy
    # input feature: [B, (2*fov_w+1)*(2*fov_h+1)]
    input_features: int = (2 * field_of_view_height + 1) * (2 * field_of_view_width + 1)

    intermedia_features1: int = 500
    intermedia_features2: int = 300
    intermedia_features3: int = 100

    # output feature: [B, 1] -> action_idx
    output_features: int = 1

    # Policy sampling
    top_k: int = 5

    # GRPO policy training
    # Train / Eval / Test
    lr = 1.0
    epoches: int = 2
    episode_group_size: int = 50
    episode_steps: int = 2
    eval_steps: int = 2
    # episodes_per_iteration: int = 2

    # EPSILON
    epsilon = 1e-5

    # Device
    device = torch.device("cpu")  # torch.device("mps")

    # Figure size
    figure_size: tuple[int, int] = (5, 5)
    double_figure_size: tuple[int, int] = (10, 5)

    # Rewards
    max_reward = torch.tensor(500.0)
    blocked_reward = torch.tensor(-500.0)
