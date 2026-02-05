import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from core.rl_base import RLAgent
from core.utils.replay_buffer import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(64 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) # Continuous action space [-1, 1]

class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Action concatenated at FC layer
        self.fc1 = nn.Linear(64 * 9 * 9 + action_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DDPGAgent(RLAgent):
    """
    Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning.
    """
    def __init__(self, config):
        super(DDPGAgent, self).__init__(config)
        self.action_dim = config.get("action_dim")
        self.device = config.get("device", "cpu")
        self.tau = config.get("tau", 0.005) # Soft update parameter
        self.batch_size = config.get("batch_size", 64)

        self.actor = Actor(self.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.get("actor_lr", 1e-4))

        self.critic = Critic(self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 1e-3))

        self.memory = ReplayBuffer(config.get("buffer_capacity", 50000))

    def select_action(self, state, evaluation=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        
        if not evaluation:
            # Simple noise for exploration
            action += np.random.normal(0, 0.1, size=self.action_dim)
            
        return np.clip(action, -1, 1)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update Targets
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def forward(self, x):
        return self.actor(x)

    def compute_loss(self, *args):
        return torch.tensor(0.0)
