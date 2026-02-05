import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.rl_base import RLAgent

class SARSAQNetwork(nn.Module):
    """Atari 이미지 입력을 처리하기 위한 CNN 구조 (SARSA용)"""
    def __init__(self, action_dim):
        super(SARSAQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SARSAAgent(RLAgent):
    """
    Rummery, G. A., & Niranjan, M. (1994). 
    Online Q-Learning using Connectionist Systems (SARSA).
    """
    def __init__(self, config):
        super(SARSAAgent, self).__init__(config)
        self.action_dim = config.get("action_dim")
        self.lr = config.get("lr", 0.00025)
        self.device = config.get("device", "cpu")
        
        self.q_net = SARSAQNetwork(self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state, evaluation=False):
        if not evaluation and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return torch.argmax(q_values).item()

    def forward(self, x):
        return self.q_net(x)

    def compute_loss(self, state, action, reward, next_state, next_action, done):
        """
        SARSA Update Rule (On-policy):
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        current_q = self.q_net(state_t)[0, action]
        
        with torch.no_grad():
            # SARSA uses the actual next action chosen by the policy
            next_q = self.q_net(next_state_t)[0, next_action]
            target_q = reward + (1 - done) * self.gamma * next_q
            
        return self.criterion(current_q, target_q)

    def update(self, state, action, reward, next_state, next_action, done):
        self.optimizer.zero_grad()
        loss = self.compute_loss(state, action, reward, next_state, next_action, done)
        loss.backward()
        self.optimizer.step()
        return loss.item()
