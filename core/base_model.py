import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
