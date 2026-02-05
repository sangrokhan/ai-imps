import torch
import yaml
import argparse
from core.trainer import Trainer
from models.implementation.apd_iclr2020.model import APDModel

def main(args):
    # 설정 파일 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 데이터셋 로드 (APD는 Continual Learning이므로 task_id별 로딩 필요)
    # TODO: datasets/continual_loader.py 구현 필요
    print("Loading datasets...")

    # 2. 모델 초기화
    if config.get("model_type") == "APD":
        model = APDModel(config)
    elif config.get("model_type") == "DEN":
        from models.den_model import DENModel
        model = DENModel(config)
    else:
        print(f"Unknown model type: {config.get('model_type')}")
        return

    print(f"Initializing {config.get('model_type')} model...")

    # 3. 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))

    # 4. 트레이너 실행
    # APD 특화 트레이너가 필요할 수 있으나 우선 기본 트레이너 사용
    # trainer = Trainer(model, train_loader, val_loader, optimizer, device, config)
    print("Framework integration complete for APD.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args)
