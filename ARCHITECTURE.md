# AI Implementation Framework (AI-IMPS) Architecture

This document defines the modular architecture for implementing various AI research papers (Supervised Learning, Reinforcement Learning, etc.) within the `ai-imps` repository.

## 1. Design Principles
- **Modularity**: Decouple data, models, and training logic.
- **Extensibility**: Easy to add new papers by inheriting base classes.
- **Consistency**: Unified logging, checkpointing, and configuration management.
- **Agent-Friendly**: Clear structure and documentation for AI coding agents to follow.

## 2. Directory Structure

```text
~/repo/ai-imps/
├── core/                 # Core abstractions (Interfaces)
│   ├── base_model.py     # Abstract class for all models
│   ├── base_runner.py    # Training/Interaction loop logic
│   ├── base_data.py      # Data loading abstractions
│   └── registry.py       # Dynamic component registration
├── common/               # Shared utilities and modules
│   ├── layers/           # Common neural network layers
│   ├── losses/           # Custom loss functions
│   ├── metrics/          # Evaluation metrics
│   └── utils/            # Logging, IO, etc.
├── runners/              # Standard execution engines
│   ├── supervised.py     # Standard SL training loop
│   └── reinforcement.py  # Standard RL interaction loop (Env-Agent)
├── implementations/      # Paper-specific implementations
│   ├── [paper_name]/     # e.g., den_iclr2018, apd_iclr2020
│   │   ├── model.py
│   │   ├── config.yaml
│   │   └── README.md
├── configs/              # Global configuration files
└── data/                 # Data storage and processing scripts
```

## 3. Core Components

### 3.1. Registry
A central system to register and retrieve classes. This allows switching between models or trainers using a simple string in a configuration file.

### 3.2. BaseRunner
The `Runner` is responsible for the execution flow.
- **SupervisedRunner**: Handles `(data -> model -> loss -> optimizer)`.
- **ReinforcementRunner**: Handles `(agent <-> environment -> reward -> update)`.

### 3.3. Config Management
Use a unified configuration format (YAML or Hydra) to define hyperparameters, paths, and component selections.

## 4. Workflow for New Paper Implementation
1. Create a subfolder in `implementations/`.
2. Define the model by inheriting from `core.base_model.BaseModel`.
3. If it requires a custom training loop, inherit from `core.base_runner.BaseRunner`.
4. Create a configuration file.
5. Execute using a global `main.py` entry point that uses the `Registry`.
