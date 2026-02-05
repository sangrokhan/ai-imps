import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core.base_model import BaseModel

class LoRALinear(nn.Module):
    """Low-Rank Adaptation이 적용된 선형 레이어"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super(LoRALinear, self).__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        # Original Weights (Pre-trained)
        self.pretrained_w = nn.Linear(in_features, out_features)
        self.pretrained_w.weight.requires_grad = False # Freeze
        
        # LoRA Matrices (Trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Base Path
        base_out = self.pretrained_w(x)
        # LoRA Path: (x @ A.T @ B.T)
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out

import math # Kaiming init을 위해 추가

class LoRAModel(BaseModel):
    """
    LoRA: Low-Rank Adaptation of Large Language Models (2021)
    """
    def __init__(self, config):
        super(LoRAModel, self).__init__(config)
        self.input_dim = config.get("input_dim", 784)
        self.hidden_dim = config.get("hidden_dim", 400)
        self.output_dim = config.get("output_dim", 10)
        self.rank = config.get("rank", 8)
        
        self.layer1 = LoRALinear(self.input_dim, self.hidden_dim, rank=self.rank)
        self.layer2 = LoRALinear(self.hidden_dim, self.output_dim, rank=self.rank)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

    def compute_loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)
