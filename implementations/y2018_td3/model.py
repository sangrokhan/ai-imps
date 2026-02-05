import torch
import torch.nn as nn
import torch.nn.functional as F
from core.rl_base import RLAgent
from core.utils.replay_buffer import ReplayBuffer

class TD3Network(nn.Module):
    def __init__(self, action_dim):
        super(TD3Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU()
        )
        self.actor = nn.Sequential(nn.Linear(64 * 9 * 9, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Tanh())
        self.q1 = nn.Sequential(nn.Linear(64 * 9 * 9 + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(64 * 9 * 9 + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.conv(x).view(x.size(0), -1)

class TD3Agent(RLAgent):
    """Fujimoto et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods."""
    def __init__(self, config):
        super(TD3Agent, self).__init__(config)
        self.action_dim = config.get("action_dim")
        self.device = config.get("device", "cpu")
        self.model = TD3Network(self.action_dim).to(self.device)
        self.memory = ReplayBuffer(config.get("buffer_capacity", 50000))

    def select_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        feat = self.model(state)
        action = self.model.actor(feat).cpu().detach().numpy()[0]
        return action

    def update(self): pass
    def compute_loss(self, *args): return torch.tensor(0.0)
    def forward(self, x): return self.model(x)
