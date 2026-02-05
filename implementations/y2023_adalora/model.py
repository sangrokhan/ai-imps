import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel

class AdaLoRALinear(nn.Module):
    """Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (2023)"""
    def __init__(self, in_features, out_features, rank=8):
        super(AdaLoRALinear, self).__init__()
        self.rank = rank
        
        # Frozen Weights
        self.pretrained_w = nn.Linear(in_features, out_features)
        self.pretrained_w.weight.requires_grad = False
        
        # SVD-based LoRA Matrices: AB -> P * Lambda * Q
        self.lora_P = nn.Parameter(torch.randn(out_features, rank))
        self.lora_Lambda = nn.Parameter(torch.zeros(rank))
        self.lora_Q = nn.Parameter(torch.randn(rank, in_features))

    def forward(self, x):
        base_out = self.pretrained_w(x)
        # Adaptive Low-rank out
        # Lambda를 Diagonal 행렬로 취급하여 순차적 행렬 곱
        lora_out = x @ self.lora_Q.t() @ torch.diag(self.lora_Lambda) @ self.lora_P.t()
        return base_out + lora_out

class AdaLoRAModel(BaseModel):
    """
    AdaLoRA: SVD 형태의 분해를 통해 중요도에 따라 랭크 예산을 동적으로 할당.
    """
    def __init__(self, config):
        super(AdaLoRAModel, self).__init__(config)
        self.input_dim = config.get("input_dim", 784)
        self.hidden_dim = config.get("hidden_dim", 400)
        self.output_dim = config.get("output_dim", 10)
        
        self.layer1 = AdaLoRALinear(self.input_dim, self.hidden_dim)
        self.layer2 = AdaLoRALinear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

    def compute_loss(self, outputs, targets):
        """기본 손실 함수 + SVD 직교성 유지를 위한 Orthogonality Penalty(논문 핵심)"""
        base_loss = F.cross_entropy(outputs, targets)
        
        # Orthogonality Penalty (P.T @ P - I)
        penalty = torch.norm(self.layer1.lora_P.t() @ self.layer1.lora_P - torch.eye(self.layer1.rank).to(outputs.device))
        
        return base_loss + 0.1 * penalty
