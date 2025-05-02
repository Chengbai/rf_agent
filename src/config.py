import torch


class Config:
    # world
    world_min_x: float = -100.0
    world_max_x: float = 100.0
    world_min_y: float = -100.0
    world_max_y: float = 100.0

    # actions
    possible_actions: list[tuple[float, float]] = [
        (1.0 * i, 1.0 * j) for i in range(-1, 2) for j in range(-1, 2)
    ]

    # policy
    input_features: int = 2
    output_features: int = 1
    intermedia_features: int = 100

    # policy sampling
    top_k: int = 5

    # GRPO policy training
    episode_steps: int = 50
    episodes_per_iteration: int = 200

    # EPSILON
    epsilon = 1e-5

    # device
    device = torch.device("cpu")
