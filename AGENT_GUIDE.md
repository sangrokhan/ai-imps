# Coding Agent Guide: Implementing New Papers

This guide provides instructions for AI agents to implement research papers within the `ai-imps` framework.

## 1. Directory Structure

All implementations must follow this structure to ensure consistency:

```text
implementations/
└── [paper_id]/         # e.g., y2015_dqn, y2017_ppo
    ├── model.py        # Implementation code (PyTorch/etc)
    ├── config.yaml     # Paper-specific hyperparameters and settings
    └── [paper_id].pdf  # Original research paper PDF
```

## 2. Preparation
Before starting a new implementation:
- Read the paper carefully to identify the core architecture, loss functions, and hyperparameters.
- Check if any components (layers, utilities) already exist in `common/`.

## 3. Implementation Steps

### Step 1: Create the Workspace
Create a folder under `implementations/[paper_id]`.
```bash
mkdir -p implementations/y2024_new_paper
```

### Step 2: Define the Model
Inherit from `core.BaseModel`. Ensure you support the standard `forward` interface.
```python
from core.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Define layers
    
    def forward(self, x):
        # Implementation
        return x
```

### Step 3: Configure Settings
Create a `config.yaml` file **within the paper's folder**.
- Define `model_name`: Used by Registry.
- Define `runner_type`: `supervised` or `reinforcement`.
- Define `hyperparameters`: learning_rate, batch_size, etc.

### Step 4: Paper Storage
Store the original paper PDF directly in the implementation folder as `[paper_id].pdf`.

### Step 5: Custom Logic (If needed)
- For **Supervised Learning**: If the loss or data loading is unique, add them to `common/` or keep them paper-specific within the implementation folder.
- For **Reinforcement Learning**: Define the environment in `implementations/[paper_id]/env.py` if it's custom.

## 4. Coding Standards
- **Docstrings**: Every class and major function must have a docstring.
- **Type Hinting**: Use Python type hints for better readability and agent debugging.
- **Logging**: Use the unified logger from `common.utils.logger`.
- **Config-Driven**: Avoid hardcoding values; always use the `config` object.

## 5. Verification
- Implement unit tests for custom layers in `tests/`.
- Run a small-scale "sanity check" training run to ensure no runtime errors.
- Compare the model output with the paper's reported values if possible.

## 6. Paper-Specific README
Always include a `README.md` in the implementation folder with:
- Link to the original paper.
- Summary of implemented features.
- Instructions to reproduce results.
