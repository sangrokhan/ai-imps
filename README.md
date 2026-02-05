# AI Implementation Framework (AI-IMPS)

A modular framework for implementing and experimenting with AI research papers.

## 📂 Project Structure
- `core/`: Abstract base classes and registry system.
- `common/`: Shared layers, loss functions, and utilities.
- `runners/`: Standard execution engines (DENRunner, Supervised, RL).
- `implementations/`: Specific paper implementations (follows `yYEAR_CONFERENCE_NAME` naming convention).
- `configs/`: Configuration management (YAML).
- `tests/`: Project integrity and unit tests.

## 🛠️ Usage

### Environment Setup
```bash
# Recommended: Using the existing venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running an Implementation
Use `main.py` with a configuration file:
```bash
python main.py --config configs/2018_iclr_den.yaml
python main.py --config configs/2020_iclr_apd.yaml
```

### Running Tests
Always run tests before committing or reporting task completion:
```bash
python -m pytest tests/
```

## 📜 Development Guidelines (for AI Agents)
1. **Repository Scope**: All development must happen within `~/repo/ai-imps`.
2. **Test-Driven**: Create unit tests in `tests/` for new features or bug fixes.
3. **No Deletion**: Never delete or comment out failing test cases. Fix the code instead.
4. **Integrity Check**: Ensure all tests pass via `pytest` before reporting completion.
