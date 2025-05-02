#!/usr/bin/env python3
# Navigation RL Agent - DQN Tests
# Author: Kashad J. Turner‑Warren

import unittest
import numpy as np
import torch
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dqn_agent import QNetwork, ReplayBuffer, DQNAgent
from src.environment_wrapper import BananaEnv

class TestReplayBuffer(unittest.TestCase):
    """Test cases for the ReplayBuffer class."""
    
    def setUp(self):
        """Set up the test fixture."""
        self.action_size = 4
        self.buffer_size = 100
        self.batch_size = 5
        self.seed = 0
        self.buffer = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        
        # Add some experiences to the buffer
        for i in range(10):
            state = np.array([i, i+1, i+2, i+3])
            action = i % self.action_size
            reward = float(i)
            next_state = np.array([i+1, i+2, i+3, i+4])
            done = bool(i % 2)
            self.buffer.add(state, action, reward, next_state, done)
    
    def test_buffer_initialization(self):
        """Test that the buffer is initialized correctly."""
        self.assertEqual(self.buffer.action_size, self.action_size)
        self.assertEqual(self.buffer.batch_size, self.batch_size)
        self.assertEqual(len(self.buffer), 10)
    
    def test_buffer_capacity(self):
        """Test that the buffer respects its capacity."""
        # Add more experiences than the buffer capacity
        for i in range(self.buffer_size):
            state = np.array([i, i+1, i+2, i+3])
            action = i % self.action_size
            reward = float(i)
            next_state = np.array([i+1, i+2, i+3, i+4])
            done = bool(i % 2)
            self.buffer.add(state, action, reward, next_state, done)
        
        # Check that the buffer size is equal to its capacity
        self.assertEqual(len(self.buffer), self.buffer_size)
    
    def test_sample(self):
        """Test that sampling from the buffer works correctly."""
        # Sample from the buffer
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        # Check the shapes of the returned tensors
        self.assertEqual(states.shape, (self.batch_size, 4))
        self.assertEqual(actions.shape, (self.batch_size, 1))
        self.assertEqual(rewards.shape, (self.batch_size, 1))
        self.assertEqual(next_states.shape, (self.batch_size, 4))
        self.assertEqual(dones.shape, (self.batch_size, 1))
        
        # Check that the types are correct
        self.assertEqual(states.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.int64)
        self.assertEqual(rewards.dtype, torch.float32)
        self.assertEqual(next_states.dtype, torch.float32)
        self.assertEqual(dones.dtype, torch.float32)


class TestQNetwork(unittest.TestCase):
    """Test cases for the QNetwork class."""
    
    def setUp(self):
        """Set up the test fixture."""
        self.state_size = 4
        self.action_size = 4
        self.seed = 0
        self.batch_size = 5
        self.network = QNetwork(self.state_size, self.action_size, self.seed)
    
    def test_network_initialization(self):
        """Test that the network is initialized correctly."""
        # Check that the network has the correct number of layers
        self.assertEqual(len(list(self.network.parameters())), 6)  # 3 weight matrices and 3 bias vectors
    
    def test_forward_pass(self):
        """Test that the forward pass works correctly."""
        # Create a batch of states
        states = torch.randn(self.batch_size, self.state_size)
        
        # Perform a forward pass
        action_values = self.network(states)
        
        # Check the shape of the output
        self.assertEqual(action_values.shape, (self.batch_size, self.action_size))


class TestDQNAgent(unittest.TestCase):
    """Test cases for the DQNAgent class."""
    
    def setUp(self):
        """Set up the test fixture."""
        self.state_size = 4
        self.action_size = 4
        self.seed = 0
        self.agent = DQNAgent(self.state_size, self.action_size, self.seed)
    
    def test_agent_initialization(self):
        """Test that the agent is initialized correctly."""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.eps, 1.0)
    
    def test_action_selection(self):
        """Test that action selection works correctly."""
        # Create a state
        state = np.random.rand(self.state_size)
        
        # Get an action with epsilon=0 (greedy)
        action = self.agent.act(state, eps=0)
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_size)
        
        # Get an action with epsilon=1 (random)
        action = self.agent.act(state, eps=1)
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_size)
    
    def test_learning(self):
        """Test that learning works correctly."""
        # Create a batch of experiences
        states = np.random.rand(5, self.state_size)
        actions = np.random.randint(0, self.action_size, (5, 1))
        rewards = np.random.rand(5, 1)
        next_states = np.random.rand(5, self.state_size)
        dones = np.random.randint(0, 2, (5, 1)).astype(np.uint8)
        
        # Convert to tensors
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float()
        
        # Create experiences tuple
        experiences = (states, actions, rewards, next_states, dones)
        
        # Learn from experiences
        self.agent.learn(experiences, self.agent.gamma)
        
        # Check that epsilon was updated
        self.assertLess(self.agent.eps, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration test for the DQN agent and environment."""
    
    def test_single_training_iteration(self):
        """Test that a single training iteration works without errors."""
        # Create a mock environment
        class MockEnv:
            def __init__(self):
                self.state_size = 4
                self.action_size = 4
            
            def reset(self):
                return np.random.rand(self.state_size)
            
            def step(self, action):
                next_state = np.random.rand(self.state_size)
                reward = float(np.random.rand())
                done = bool(np.random.randint(0, 2))
                return next_state, reward, done, {}
        
        env = MockEnv()
        
        # Create an agent
        agent = DQNAgent(env.state_size, env.action_size, seed=0)
        
        # Run a single episode
        state = env.reset()
        for _ in range(10):  # Take 10 steps
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break


if __name__ == '__main__':
    unittest.main()