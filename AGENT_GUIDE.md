# Coding Agent Guide: Implementing New Papers

This guide provides instructions for AI agents to implement research papers within the `ai-imps` framework.

## 1. Preparation
Before starting a new implementation:
- Read the paper carefully to identify the core architecture, loss functions, and hyperparameters.
- Check if any components (layers, utilities) already exist in `common/`.

## 2. Implementation Steps

### Step 1: Create the Workspace
Create a folder under `implementations/[paper_id]`.
```bash
mkdir -p implementations/my_new_paper
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

### Step 3: Configure Training/Interaction
Define a `config.yaml` specifying:
- `model_name`: Used by Registry.
- `runner_type`: `supervised` or `reinforcement`.
- `hyperparameters`: learning_rate, batch_size, etc.

### Step 4: Custom Logic (If needed)
- For **Supervised Learning**: If the loss or data loading is unique, add them to `common/` or keep them paper-specific within the implementation folder.
- For **Reinforcement Learning**: Define the environment in `implementations/[paper_id]/env.py` if it's custom.

## 3. Coding Standards
- **Docstrings**: Every class and major function must have a docstring.
- **Type Hinting**: Use Python type hints for better readability and agent debugging.
- **Logging**: Use the unified logger from `common.utils.logger`.
- **Config-Driven**: Avoid hardcoding values; always use the `config` object.

## 4. Verification
- Implement unit tests for custom layers in `tests/`.
- Run a small-scale "sanity check" training run to ensure no runtime errors.
- Compare the model output with the paper's reported values if possible.

## 5. Paper-Specific README
Always include a `README.md` in the implementation folder with:
- Link to the original paper.
- Summary of implemented features.
- Instructions to reproduce results.
