import pytest
import torch
import numpy as np
from implementations.q_learning_1992.model import QLearningAgent
from implementations.y1992_reinforce.model import REINFORCEAgent
from implementations.y1994_sarsa.model import SARSAAgent

@pytest.fixture
def rl_config():
    return {
        "action_dim": 4,
        "lr": 0.001,
        "device": "cpu",
        "gamma": 0.99
    }

def test_q_learning_initialization(rl_config):
    agent = QLearningAgent(rl_config)
    assert agent.action_dim == 4
    assert isinstance(agent.q_net, torch.nn.Module)

def test_reinforce_initialization(rl_config):
    agent = REINFORCEAgent(rl_config)
    assert agent.action_dim == 4
    assert isinstance(agent.policy_net, torch.nn.Module)

def test_sarsa_initialization(rl_config):
    agent = SARSAAgent(rl_config)
    assert agent.action_dim == 4
    assert isinstance(agent.q_net, torch.nn.Module)

def test_q_learning_action_selection(rl_config):
    agent = QLearningAgent(rl_config)
    agent.epsilon = 0.0 # Force greedy
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action < 4

def test_reinforce_action_selection(rl_config):
    agent = REINFORCEAgent(rl_config)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action < 4
    assert len(agent.saved_log_probs) == 1

def test_sarsa_action_selection(rl_config):
    agent = SARSAAgent(rl_config)
    agent.epsilon = 0.0
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = agent.select_action(state)
    assert 0 <= action < 4
