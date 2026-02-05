import time
import numpy as np
import gymnasium as gym
from core.logger import ExperimentLogger

class RLTrainer:
    """
    강화학습 에이전트의 학습을 관리하는 트레이너 클래스입니다.
    """
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = ExperimentLogger(config)
        self.device = config.get("device", "cpu")
        
        self.total_steps = 0
        self.best_reward = -float('inf')

    def run_episode(self, evaluation=False):
        state, _ = self.env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = self.agent.select_action(state, evaluation=evaluation)
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            if not evaluation:
                loss = self.agent.update(state, action, reward, next_state, done)
                self.total_steps += 1
                if self.total_steps % self.config.get("log_interval", 100) == 0:
                    self.logger.log_scalar("Train/StepLoss", loss, self.total_steps)
            
            state = next_state
            episode_reward += reward
            
        return episode_reward

    def fit(self):
        num_episodes = self.config.get("num_episodes", 1000)
        start_time = time.time()
        
        for episode in range(num_episodes):
            reward = self.run_episode(evaluation=False)
            self.agent.decay_epsilon()
            
            # Logging
            self.logger.log_metrics({"Reward": reward, "Epsilon": self.agent.epsilon}, episode, mode="Episode")
            
            if reward > self.best_reward:
                self.best_reward = reward
                # Best model save logic...
                
            if (episode + 1) % self.config.get("eval_freq", 10) == 0:
                eval_reward = self.run_episode(evaluation=True)
                self.logger.log_scalar("Eval/Reward", eval_reward, episode)
                print(f"Episode {episode} | Eval Reward: {eval_reward:.2f} | Epsilon: {self.agent.epsilon:.3f}")

        self.logger.close()
        print(f"Training finished in {time.time() - start_time:.2f} seconds.")
