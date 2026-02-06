import argparse
import yaml
import os
import torch
from core.registry import MODEL_REGISTRY, RUNNER_REGISTRY
from core.utils.setup import set_seed, get_device
import implementations # Trigger registration
import runners         # Trigger registration

def merge_configs(base, override):
    """Deep merge two dictionaries."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def main():
    parser = argparse.ArgumentParser(description="AI-IMPS Training Entry Point")
    parser.add_argument("--config", type=str, required=True, help="Path to paper-specific config yaml")
    args = parser.parse_args()

    # 1. Load default config
    root_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(root_dir, "configs", "default.yaml")
    with open(default_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Merge with paper-specific config
    with open(args.config, 'r') as f:
        paper_config = yaml.safe_load(f)
    config = merge_configs(config, paper_config)

    # 3. Setup environment (Seed & Device)
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    config["device"] = str(device) # Ensure it's a string for child configs if needed
    
    print(f"Using device: {device}")
    print(f"Random seed set to: {config.get('seed')}")

    # 4. Initialize Model
    # Paper configs might have 'model' block or flat structure. 
    # Let's handle both for backward compatibility.
    model_cfg = config.get('model', config)
    model_name = model_cfg.get('name') or config.get('model_name')
    
    if not model_name:
        raise ValueError("Model name not specified in config.")
        
    model_cls = MODEL_REGISTRY.get(model_name)
    model = model_cls(model_cfg).to(device)
    
    print(f"Loaded model: {model_name}")

    # 5. Initialize Runner
    runner_cfg = config.get('runner', config)
    runner_name = runner_cfg.get('name') or config.get('runner_type')
    
    if not runner_name:
        raise ValueError("Runner name/type not specified in config.")

    runner_cls = RUNNER_REGISTRY.get(runner_name)
    runner = runner_cls(runner_cfg, model)
    
    print(f"Loaded runner: {runner_name}")

    # 6. Run
    runner.run()

if __name__ == "__main__":
    main()
