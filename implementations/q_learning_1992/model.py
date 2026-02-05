import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.rl_base import RLAgent

class QNetwork(nn.Module):
    """Atari 이미지 입력을 처리하기 위한 간단한 CNN 구조"""
    def __init__(self, action_dim):
        super(QNetwork, self).__init__()
        # 입력: (Batch, 4, 84, 84) - Frame Stacking 가정
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

class QLearningAgent(RLAgent):
    """
    Watkins, C. J., & Dayan, P. (1992). Technical Note: Q-Learning.
    논문의 로직을 신경망 함수 근사자로 구현한 에이전트입니다.
    """
    def __init__(self, config):
        super(QLearningAgent, self).__init__(config)
        self.action_dim = config.get("action_dim")
        self.lr = config.get("lr", 0.00025)
        
        # Q-Network 초기화
        self.q_net = QNetwork(self.action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state, evaluation=False):
        """Epsilon-greedy 액션 선택"""
        if not evaluation and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.get("device", "cpu"))
        with torch.no_grad():
            q_values = self.q_net(state)
        return torch.argmax(q_values).item()

    def forward(self, x):
        return self.q_net(x)

    def compute_loss(self, state, action, reward, next_state, done):
        """
        Q-Learning Update Rule:
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.get("device", "cpu"))
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.config.get("device", "cpu"))
        
        # Current Q-value
        current_q = self.q_net(state)[0, action]
        
        # Target Q-value
        with torch.no_grad():
            next_q_max = torch.max(self.q_net(next_state))
            target_q = reward + (1 - done) * self.gamma * next_q_max
            
        loss = self.criterion(current_q, target_q)
        return loss

    def update(self, state, action, reward, next_state, done):
        self.optimizer.zero_grad()
        loss = self.compute_loss(state, action, reward, next_state, done)
        loss.backward()
        self.optimizer.step()
        return loss.item()
