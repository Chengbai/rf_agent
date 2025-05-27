# Project: Predict the trajectory in an environment with BLOCKs.
----
- **Env**: N x M (rows x columns) grid with random generated BLOCKs
- **Inputs**:
  - start position: (x0, y0)
  - target position: (xt, yt)
- **Rules**:
  - ONLY allow move within the grid
  - CANNOT move to the BLOCK position
  - At any time, possible moves are:
     [top-left, top, top-right,
      left, current, right,
      bottom-left, bottom, bottom-right]
- **Example Setup**:
<p align="center">
    <img src="doc/example_setup1.png" alt="Description" width="200">
</p>

# Brainstorm 
---
![CoT](doc/brainstorm20250508.jpg)

# Project Architecture
---
## RF with GRPO
![RF GRPO Architecture](doc/RF_GRPO.drawio.svg)

## Policy Model
----
### Linear Models
2 models here: 
 - linear model
 - linear model with late position fusion
<p align="center">
    <img src="doc/RF_GRPO_Grid_Move_Prediction_Linear_Model.drawio.svg" alt="RF GRPO Grid Move Prediction Linear Models" width="500">
</p>

## Example Solution
<p align="center">
    <img src="doc/example_solution1.png" alt="Description" width="600">
</p>