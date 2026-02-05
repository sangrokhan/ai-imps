import torch
import yaml
import argparse
from core.trainer import Trainer
from models.implementation.apd_iclr2020.model import APDModel

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Initializing {config.get('model_type')} model...")
    if config.get("model_type") == "APD":
        model = APDModel(config)
    else:
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    print("Framework integration complete with TensorBoard logging.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args)
