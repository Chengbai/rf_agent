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

### Example Solution
<p align="center">
    <img src="doc/example_solution1.png" alt="Description" width="600">
</p>

____
### Transformer w/ Late Fusion Model

<p align="center">
    <img src="doc/Transformer.drawio.svg" alt="RF GRPO Grid Move Prediction Transformer Model" width="500">
</p>

### Example Solution

* Example from the evaluation result (topk=1) **The model learned move around the blocks**
<p align="center">
    <img src="doc/example_solution_transformer2-1.png" alt="TopK=1 (Eval)" width="600">
</p>

* Example from the test result (topk=1) **The model learned move around the blocks**
<p align="center">
    <img src="doc/example_solution_transformer2-3.png" alt="Topk=1 (Test)" width="600">
</p>

* Example from the test result (topk=2). **The model learned move around the blocks**
<p align="center">
    <img src="doc/example_solution_transformer2-2.png" alt="Topk=2 (Test)" width="600">
</p>

## GRPO-Cheng Trainer
---
<p align="center">
    <img src="doc/RL_GRPO Grid Move Trainer.drawio.svg" alt="GROP-Cheng Trainer" width="600">
</p>