# AI Implementation Framework (AI-IMPS)

A modular and scalable framework for implementing and experimenting with AI research papers across Supervised and Reinforcement Learning.

## 📂 Project Directory Structure

The project follows a modular architecture designed for consistency and extensibility:

```text
~/repo/ai-imps/
├── core/                 # Core abstractions (BaseModel, BaseRunner)
├── common/               # Shared utilities (layers, losses, metrics, loggers)
├── runners/              # Standard execution engines (supervised, reinforcement)
├── implementations/      # Paper-specific implementations (Self-contained)
│   └── [paper_id]/       # e.g., y2015_dqn, y2017_ppo
│       ├── model.py      # Core implementation code
│       ├── config.yaml   # Paper-specific hyperparameters
│       ├── runner.py     # (Optional) Custom training logic
│       └── [paper_id].pdf # Original research paper PDF
├── configs/              # Global configuration templates
│   ├── default.yaml      # Project-wide defaults (device, seed, paths)
│   ├── base_rl.yaml      # RL-specific defaults
│   └── base_supervised.yaml # SL-specific defaults
├── data/                 # Data storage
├── tests/                # Unit and integration tests
└── main.py               # Central entry point
```

## 🛠️ Getting Started

### 1. Environment Setup
We recommend using a virtual environment. If it doesn't exist, create one:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation
Download required datasets before running experiments:
```bash
python3 scripts/download_data.py --dir ./data
```

### 3. Running an Experiment
All experiments are executed via `main.py` using a paper-specific configuration file. The framework automatically handles device detection (CUDA/MPS/CPU) and seed management.

```bash
# Example: Running Deep Q-Learning (2015)
python3 main.py --config implementations/y2015_dqn/config.yaml

# Example: Running PPO (2017)
python3 main.py --config implementations/y2017_ppo/config.yaml
```

### 4. Monitoring with TensorBoard
Training progress, including loss curves and metrics, is logged to the `outputs/` directory.

```bash
tensorboard --logdir outputs/
```
Then open `http://localhost:6006` in your browser.

## ✅ Quality Assurance

### Running Tests
To ensure framework integrity, run the test suite using `pytest`:
```bash
python3 -m pytest tests/
```

## 📜 Development Guidelines (for AI Agents)
Please refer to [AGENT_GUIDE.md](AGENT_GUIDE.md) for detailed instructions on how to contribute new paper implementations, coding standards, and the PR process.
