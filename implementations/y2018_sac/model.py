import torch
import torch.nn as nn
import torch.nn.functional as F
from core.rl_base import RLAgent
from core.utils.replay_buffer import ReplayBuffer
import copy

class SACNetwork(nn.Module):
    def __init__(self, action_dim):
        super(SACNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU()
        )
        self.fc = nn.Linear(64 * 9 * 9, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.q1 = nn.Sequential(nn.Linear(64 * 9 * 9 + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(64 * 9 * 9 + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.mu(x), self.log_std(x)

class SACAgent(RLAgent):
    """Haarnoja et al. (2018). Soft Actor-Critic."""
    def __init__(self, config):
        super(SACAgent, self).__init__(config)
        self.action_dim = config.get("action_dim")
        self.device = config.get("device", "cpu")
        self.model = SACNetwork(self.action_dim).to(self.device)
        self.memory = ReplayBuffer(config.get("buffer_capacity", 50000))
    
    def select_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, log_std = self.model(state)
        if evaluation: return torch.tanh(mu).cpu().numpy()[0]
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        return torch.tanh(dist.sample()).cpu().numpy()[0]

    def update(self): pass
    def compute_loss(self, *args): return torch.tensor(0.0)
    def forward(self, x): return self.model(x)
