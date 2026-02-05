import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel

class CQLModel(BaseModel):
    """
    Conservative Q-Learning for Offline Reinforcement Learning (2020)
    데이터 분포 밖의 액션에 대해 낮은 Q-value를 강제하는 Offline RL 알고리즘.
    """
    def __init__(self, config):
        super(CQLModel, self).__init__(config)
        self.state_dim = config.get("state_dim", 128)
        self.action_dim = config.get("action_dim", 4)
        self.alpha = config.get("cql_alpha", 1.0) # CQL 패널티 계수
        
        # Q-Network
        self.q_net = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )

    def forward(self, x):
        return self.q_net(x)

    def compute_loss(self, state, action, reward, next_state, done, target_q_net, gamma):
        """
        CQL Loss = Bellman Error + alpha * (log_sum_exp(Q) - data_Q)
        """
        q_values = self.forward(state)
        current_q = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # 1. Standard Bellman Error (TD Loss)
        with torch.no_grad():
            next_q = target_q_net(next_state).max(1)[0]
            target_q = reward + (1 - done) * gamma * next_q
        
        td_loss = F.mse_loss(current_q, target_q)
        
        # 2. Conservative Penalty (CQL Term)
        # logsumexp를 통해 전체 액션 공간에 대한 Q값을 패널티화
        cql_penalty = torch.logsumexp(q_values, dim=1).mean() - current_q.mean()
        
        return td_loss + self.alpha * cql_penalty
