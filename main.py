import argparse
import yaml
import torch
from core.registry import MODEL_REGISTRY, RUNNER_REGISTRY
import implementations # Trigger registration
import runners         # Trigger registration

def main():
    parser = argparse.ArgumentParser(description="AI-IMPS Training Entry Point")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Initialize Model
    model_name = config['model']['name']
    model_cls = MODEL_REGISTRY.get(model_name)
    model = model_cls(config['model'])
    
    print(f"Loaded model: {model_name}")

    # 2. Initialize Runner
    runner_name = config['runner']['name']
    runner_cls = RUNNER_REGISTRY.get(runner_name)
    runner = runner_cls(config['runner'], model)
    
    print(f"Loaded runner: {runner_name}")

    # 3. Run
    runner.run()

if __name__ == "__main__":
    main()
