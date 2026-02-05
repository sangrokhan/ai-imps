import pytest
import torch
import numpy as np
import copy # Added for PPOAgent
from implementations.q_learning_1992.model import QLearningAgent
from implementations.y1992_reinforce.model import REINFORCEAgent
from implementations.y1994_sarsa.model import SARSAAgent
from implementations.y2015_dqn.model import DQNAgent
from implementations.y2015_ddpg.model import DDPGAgent
from implementations.y2017_ppo.model import PPOAgent
from implementations.y2018_sac.model import SACAgent
from implementations.y2018_td3.model import TD3Agent

@pytest.fixture
def rl_config():
    return {
        "action_dim": 4,
        "lr": 0.001,
        "device": "cpu",
        "gamma": 0.99,
        "batch_size": 2,
        "buffer_capacity": 10
    }

def test_q_learning(rl_config):
    agent = QLearningAgent(rl_config)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action < rl_config["action_dim"]

def test_reinforce(rl_config):
    agent = REINFORCEAgent(rl_config)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action < rl_config["action_dim"]

def test_sarsa(rl_config):
    agent = SARSAAgent(rl_config)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action < rl_config["action_dim"]

def test_dqn(rl_config):
    agent = DQNAgent(rl_config)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action < rl_config["action_dim"]
    next_state = np.random.randn(4, 84, 84).astype(np.float32)
    agent.memory.push(state, action, 1.0, next_state, False)
    agent.memory.push(state, action, 1.0, next_state, False)
    loss = agent.update()
    assert loss is not None

def test_ddpg(rl_config):
    agent = DDPGAgent(rl_config)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert action.shape == (rl_config["action_dim"],)
    next_state = np.random.randn(4, 84, 84).astype(np.float32)
    agent.memory.push(state, action, 1.0, next_state, False)
    agent.memory.push(state, action, 1.0, next_state, False)
    losses = agent.update()
    assert "actor_loss" in losses
    assert "critic_loss" in losses

def test_ppo(rl_config):
    agent = PPOAgent(rl_config)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action < rl_config["action_dim"]

def test_sac(rl_config):
    agent = SACAgent(rl_config)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert action.shape == (rl_config["action_dim"],)

def test_td3(rl_config):
    agent = TD3Agent(rl_config)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert action.shape == (rl_config["action_dim"],)
