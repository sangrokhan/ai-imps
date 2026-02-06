import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from core.rl_base import RLAgent
from core.registry import MODEL_REGISTRY
from core.utils.replay_buffer import ReplayBuffer

class DQNNetwork(nn.Module):
    def __init__(self, action_dim):
        super(DQNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

@MODEL_REGISTRY.register("dqn")
class DQNAgent(RLAgent):
    """
    Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
    """
    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
        self.action_dim = config.get("action_dim")
        self.lr = config.get("lr", 0.00025)
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.target_update_freq = config.get("target_update_freq", 1000)
        self.update_counter = 0

        self.q_net = DQNNetwork(self.action_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(config.get("buffer_capacity", 10000))

    def select_action(self, state, evaluation=False):
        if not evaluation and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return torch.argmax(q_values).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q_max
            
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        return loss.item()

    def forward(self, x):
        return self.q_net(x)

    def compute_loss(self, *args):
        # Implementation is inside update for DQN to handle sampling
        return torch.tensor(0.0)
