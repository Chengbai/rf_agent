# Create 2025
import math
import torch

from dataclasses import dataclass, field
from matplotlib.colors import ListedColormap


@dataclass
class Config:
    # Device
    device = torch.device("mps")  # torch.device("mps")

    # world
    world_min_x: int = 0
    world_max_x: int = 32
    world_min_y: int = 0
    world_max_y: int = 32
    world_width: int = int(world_max_x - world_min_x)
    world_height: int = int(world_max_y - world_min_y)
    world_block_probability = 0.2

    # Dataset
    train_dataset_length: int = 50000
    train_batch_size: int = 100

    test_dataset_length: int = 10
    test_batch_size: int = 2

    eval_dataset_length: int = 500
    eval_batch_size: int = eval_dataset_length // 5

    # Actions
    possible_actions: torch.tensor = field(
        default_factory=lambda: torch.tensor(
            [(1 * i, 1 * j) for i in range(-1, 2) for j in range(-1, 2)],
            device=Config.device,
        )
    )

    # FOV
    start_center_of_fov: bool = False
    field_of_view_width: int = world_width // 2  # FOV width
    field_of_view_height: int = world_height // 2  # FOV height

    ENCODE_TARGET_POS: int = 0
    ENCODE_BLOCK: int = 100
    ENCODE_START_POS: int = 128
    ENCODE_START_STEP_IDX = 150
    ENCODE_EMPTY: int = 255

    ENCODE_NUM_EMBEEDING: int = ENCODE_EMPTY + 1
    ENCODE_EMBEEDING_DIM: int = 3

    ENCODE_COLORS = ["red", "black", "blue", "green", "white"]
    CMAP = ListedColormap(ENCODE_COLORS)

    # Policy
    # input feature: [B, (2*fov_w+1)*(2*fov_h+1)]
    input_features: int = (2 * field_of_view_height + 1) * (2 * field_of_view_width + 1)

    intermedia_features1: int = 500
    intermedia_features2: int = 300
    intermedia_features3: int = 50

    # output feature: [B, 1] -> action_idx
    output_features: int = 1

    # Policy sampling
    top_k: int = 5

    # GRPO policy training
    # Train / Eval / Test
    lr = 1.0
    epoches: int = 10
    episode_group_size: int = 8
    episode_steps: int = 20
    # episodes_per_iteration: int = 2

    # EPSILON
    epsilon = 1e-5

    # Figure size
    figure_size: tuple[int, int] = (5, 5)
    double_figure_size: tuple[int, int] = (10, 5)
    triple_figure_size: tuple[int, int] = (15, 5)

    # Rewards
    max_reward = torch.tensor(500.0)
    blocked_reward = torch.tensor(-500.0)

    # Transformer
    multi_heads: int = 10
    embedding: int = multi_heads * 5  # Note assert embedding % multi_heads == 0
    qk_projection = multi_heads * 10
    img_in_channels: int = ENCODE_EMBEEDING_DIM
    img_kernel_size: int = 3  # in pixel
    img_tokens: int = (world_height // img_kernel_size) * (
        world_width // img_kernel_size
    )
    transformer_block_layer1: int = 2 * qk_projection
    transformer_blocks = 3
    trunk_features: int = 50
