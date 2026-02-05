import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    모든 논문 구현 모델의 추상 베이스 클래스입니다.
    이 클래스를 상속받아 구체적인 모델을 구현합니다.
    """
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        """모델의 순전파 로직을 정의합니다."""
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets):
        """Loss 계산 로직을 정의합니다."""
        pass

    def save(self, path):
        """모델 가중치 저장"""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """모델 가중치 로드"""
        self.load_state_dict(torch.load(path))
