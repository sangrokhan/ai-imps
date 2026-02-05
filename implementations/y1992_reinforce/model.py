import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from core.rl_base import RLAgent

class PolicyNetwork(nn.Module):
    """Simple Policy Network for REINFORCE"""
    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__()
        # Input: (Batch, 4, 84, 84) - Atari Frame Stacking
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class REINFORCEAgent(RLAgent):
    """
    Williams, R. J. (1992). Simple statistical gradient-following algorithms 
    for connectionist reinforcement learning.
    """
    def __init__(self, config):
        super(REINFORCEAgent, self).__init__(config)
        self.action_dim = config.get("action_dim")
        self.lr = config.get("lr", 0.0005)
        
        self.device = config.get("device", "cpu")
        self.policy_net = PolicyNetwork(self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Episodic storage
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        
        if not evaluation:
            self.saved_log_probs.append(m.log_prob(action))
            
        return action.item()

    def forward(self, x):
        return self.policy_net(x)

    def compute_loss(self, state, action, reward, next_state, done):
        """
        REINFORCE computes loss based on the entire episode, 
        so this dummy method is provided to satisfy the interface.
        """
        return torch.tensor(0.0)

    def update(self):
        """Perform policy gradient update at the end of an episode"""
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate returns in reverse
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.device)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for log_prob, Gt in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * Gt)
            
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Clean up
        del self.rewards[:]
        del self.saved_log_probs[:]
        
        return loss.item()
