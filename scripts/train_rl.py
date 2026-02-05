import torch
import yaml
import argparse
from core.rl_trainer import RLTrainer
from core.env_utils import make_atari_env
from models.implementation.q_learning_1992.model import QLearningAgent

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. 환경 생성
    env_id = config.get("env_id", "PongNoFrameskip-v4")
    env = make_atari_env(env_id)
    config["action_dim"] = env.action_space.n
    
    # 2. 에이전트 초기화
    if config.get("algorithm") == "Q-Learning":
        agent = QLearningAgent(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.get('algorithm')}")
        
    # 3. 트레이너 실행
    trainer = RLTrainer(agent, env, config)
    trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rl_config.yaml")
    args = parser.parse_args()
    main(args)
