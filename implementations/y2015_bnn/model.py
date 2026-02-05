import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel

class BayesianLinear(nn.Module):
    """가중치를 분포로 취급하는 베이지안 선형 레이어"""
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        # 가중치 분포의 파라미터 (mu, rho) -> sigma = log(1 + exp(rho))
        self.mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3.0))
        
    def forward(self, x, sample=True):
        if sample:
            sigma = torch.log1p(torch.exp(self.rho))
            epsilon = torch.randn_like(sigma)
            w = self.mu + sigma * epsilon
        else:
            w = self.mu
        return F.linear(x, w)

    def kl_divergence(self):
        """Prior(Standard Normal)와의 KL Divergence 계산"""
        sigma = torch.log1p(torch.exp(self.rho))
        kl = 0.5 * (torch.pow(sigma, 2) + torch.pow(self.mu, 2) - 2 * torch.log(sigma) - 1)
        return kl.sum()

class BNNModel(BaseModel):
    """
    Weight Uncertainty in Neural Networks (2015) - Bayes by Backprop
    """
    def __init__(self, config):
        super(BNNModel, self).__init__(config)
        self.input_dim = config.get("input_dim", 784)
        self.hidden_dim = config.get("hidden_dim", 400)
        self.output_dim = config.get("output_dim", 10)
        
        self.l1 = BayesianLinear(self.input_dim, self.hidden_dim)
        self.l2 = BayesianLinear(self.hidden_dim, self.output_dim)

    def forward(self, x, sample=True):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x, sample))
        x = self.l2(x, sample)
        return x

    def compute_loss(self, outputs, targets, complexity_weight=0.01):
        """ELBO = Likelihood + Complexity Cost(KL)"""
        likelihood = F.cross_entropy(outputs, targets)
        kl = (self.l1.kl_divergence() + self.l2.kl_divergence())
        return likelihood + complexity_weight * kl
