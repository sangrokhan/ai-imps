import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from core.rl_base import RLAgent

class A3CNetwork(nn.Module):
    def __init__(self, action_dim):
        super(A3CNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU()
        )
        self.fc = nn.Linear(64 * 9 * 9, 256)
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

class A3CAgent(RLAgent):
    """Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning."""
    def __init__(self, config):
        super(A3CAgent, self).__init__(config)
        self.action_dim = config.get("action_dim")
        self.device = config.get("device", "cpu")
        self.model = A3CNetwork(self.action_dim).to(self.device)
        
    def select_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.model(state)
        m = Categorical(probs)
        return m.sample().item()

    def update(self): pass # A3C is implemented via multiple workers in trainer
    def compute_loss(self, *args): return torch.tensor(0.0)
    def forward(self, x): return self.model(x)
