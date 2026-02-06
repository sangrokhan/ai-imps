# Coding Agent Guide: Implementing New Papers

This guide defines the standardized workflow and coding standards for implementing research papers within the `ai-imps` framework.

## 1. Project Directory Structure

The project follows a strict modular structure. Agents must respect this organization:

```text
~/repo/ai-imps/
├── core/                 # Core abstractions (Interfaces)
│   ├── base_model.py     # BaseModel (Inherit for all implementations)
│   ├── base_runner.py    # BaseRunner (Inherit for custom training loops)
│   └── registry.py       # MODEL_REGISTRY, RUNNER_REGISTRY
├── common/               # Shared utilities
│   ├── layers/           # Reusable NN blocks (e.g., Attention, ResNet blocks)
│   ├── losses/           # Shared loss functions
│   └── utils/            # Logger, ReplayBuffer, IO utilities
├── runners/              # Standard execution engines
│   ├── supervised.py     # Default SL trainer
│   └── reinforcement.py  # Default RL trainer
├── implementations/      # Paper-specific implementations (Self-contained)
│   └── [paper_id]/       # e.g., y2015_dqn (year_lowercase_id)
│       ├── model.py      # Core logic and classes
│       ├── config.yaml   # Paper-specific hyperparameters
│       ├── runner.py     # Optional: Custom training logic
│       └── [paper_id].pdf # The source research paper (Put it here directly)
├── configs/              # Global configuration templates
│   ├── default.yaml      # Project-wide defaults (device, seed, paths)
│   ├── base_rl.yaml      # RL-specific defaults
│   └── base_supervised.yaml # SL-specific defaults
├── tests/                # Unit and integration tests
└── main.py               # Central entry point
```

## 2. Implementation Rules

### 2.1. Self-Containment
Every implementation must be **self-contained** within its folder. 
- **PDF**: The paper PDF must reside inside the folder.
- **Direct Path**: DO NOT create a separate "papers/" subdirectory. Place the PDF directly in the implementation folder root.
- **Config**: The `config.yaml` must define everything needed to run that specific paper.
- **Registration**: All models and runners must use `@MODEL_REGISTRY.register("name")` or `@RUNNER_REGISTRY.register("name")`.

### 2.2. Folder Naming Convention
Use `yYYYY_name` (e.g., `y2015_dqn`, `y2017_ppo`).

## 3. Workflow for Agents

### Step 1: Environment & Branching
- Always work in a separate branch (e.g., `feat/paper-name`).
- Commit frequently with descriptive messages and **Sign-off (`-s`)**.

### Step 2: Paper Analysis
- Read the PDF in `implementations/[paper_id]/`.
- Identify: Input/Output shapes, Activation functions, Loss functions, and Hyperparameters.

### Step 3: Coding the Model (`model.py`)
- Inherit from `BaseModel`.
- Register the class:
```python
@MODEL_REGISTRY.register("paper_name")
class PaperModel(BaseModel):
    ...
```

### Step 4: Configuration (`config.yaml`)
- Refer to `configs/default.yaml` and `configs/base_*.yaml`.
- Define only the values that are specific to this paper or need overriding.
- **Device & Seed**: By default, `main.py` handles `device: "auto"` (CUDA -> MPS -> CPU fallback) and sets a global random seed. Only specify `device` in your paper config if it *must* run on a specific hardware.

### Step 5: Testing & PR
- Create a test case in `tests/` or a local verification script.
- Push your changes to the remote branch immediately after implementation.
- **Create a Pull Request** so the user can review the results on GitHub.

## 4. Coding Standards

### 4.1. Avoid Hardcoding
Use the `config` object for all hyperparameters. 
- Bad: `self.lr = 0.001`
- Good: `self.lr = config.get("lr", 0.001)`

### 4.2. Logging & Error Handling
- Use `self.logger` (inherited from `BaseModel`/`BaseRunner`).
- Use `try-except` blocks for data loading or complex mathematical operations.

### 4.3. Documentation
- Use Python Type Hints for all function signatures.
- Add Docstrings following the Google style.

## 5. Verification Checklist
- [ ] Model inherits from `BaseModel`.
- [ ] Model/Runner is registered in `REGISTRY`.
- [ ] `config.yaml` is present and correctly formatted.
- [ ] Paper PDF is in the same folder.
- [ ] Code passes basic unit tests.
- [ ] Changes are pushed and PR is created.
