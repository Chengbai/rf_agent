class Config:
    # world
    world_min_x = -10
    world_max_x = 10
    world_min_y = -10
    world_max_y = 10

    # actions
    possible_actions = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
