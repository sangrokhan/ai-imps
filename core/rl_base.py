import torch
from abc import abstractmethod
from core.base_model import BaseModel

class RLAgent(BaseModel):
    """
    강화학습 에이전트를 위한 베이스 클래스입니다.
    """
    def __init__(self, config):
        super(RLAgent, self).__init__(config)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

    @abstractmethod
    def select_action(self, state):
        """상태를 입력받아 액션을 선택합니다 (Epsilon-greedy 등)."""
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """한 스텝의 경험을 바탕으로 모델을 업데이트합니다."""
        pass

    def decay_epsilon(self):
        """탐험률(Epsilon)을 감소시킵니다."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
