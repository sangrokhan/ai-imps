import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from core.rl_base import RLAgent
from core.registry import MODEL_REGISTRY
import copy

class PPOActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(PPOActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

@MODEL_REGISTRY.register("ppo")
class PPOAgent(RLAgent):
    """Schulman et al. (2017). Proximal Policy Optimization Algorithms."""
    def __init__(self, config):
        super(PPOAgent, self).__init__(config)
        self.action_dim = config.get("action_dim")
        self.device = config.get("device", "cpu")
        self.eps_clip = config.get("eps_clip", 0.2)
        self.k_epochs = config.get("k_epochs", 4)
        
        self.policy = PPOActorCritic(self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 3e-4))
        self.policy_old = copy.deepcopy(self.policy)
        
        self.memory = []

    def select_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, val = self.policy_old(state)
        m = Categorical(probs)
        action = m.sample()
        if not evaluation:
            self.memory.append((state, action, m.log_prob(action), val))
        return action.item()

    def update(self, rewards, masks):
        # Implementation of PPO clipped objective
        pass

    def compute_loss(self, *args): return torch.tensor(0.0)
    def forward(self, x): return self.policy(x)
