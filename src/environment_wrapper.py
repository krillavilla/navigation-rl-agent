#!/usr/bin/env python3
# Navigation RL Agent - Environment Wrapper
# Author: Kashad J. Turner‑Warren

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

class BananaEnv:
    """
    A wrapper for the Unity Banana Collector environment that provides a Gym-like interface.

    This wrapper handles the communication with the Unity environment and exposes a simple
    interface for reinforcement learning agents to interact with the environment.

    Attributes:
        env (UnityEnvironment): The Unity environment instance.
        behavior_name (str): The name of the behavior to use.
        action_size (int): The number of possible actions (4).
        state_size (int): The dimension of the state space (37).
        train_mode (bool): Whether to run the environment in training mode.
    """

    def __init__(self, env_path, worker_id=0, seed=0, no_graphics=False, train_mode=True):
        """
        Initialize the environment wrapper.

        Args:
            env_path (str): Path to the Unity environment executable.
            worker_id (int): Worker ID for Unity environment.
            seed (int): Random seed.
            no_graphics (bool): Whether to run the environment without graphics.
            train_mode (bool): Whether to run the environment in training mode.
        """
        # Create a channel to configure the engine
        self.engine_configuration_channel = EngineConfigurationChannel()

        # Create the environment
        self.env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            seed=seed,
            no_graphics=no_graphics,
            side_channels=[self.engine_configuration_channel]
        )

        # Set the engine configuration
        if train_mode:
            self.engine_configuration_channel.set_configuration_parameters(time_scale=20.0)
        else:
            self.engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        # Store train mode
        self.train_mode = train_mode

        # Reset the environment to get behavior names
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

        # Get action and state sizes
        self.action_size = self.spec.action_spec.discrete_size

        # Get a sample state to determine state size
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self.state_size = len(decision_steps.obs[0][0])

        print(f"Initialized Banana environment with:")
        print(f"  - State size: {self.state_size}")
        print(f"  - Action size: {self.action_size}")

    def reset(self):
        """
        Reset the environment and return the initial state.

        Returns:
            numpy.ndarray: The initial state vector.
        """
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return decision_steps.obs[0][0]

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action (int): The action to take (0-3).
                0: move forward
                1: move backward
                2: turn left
                3: turn right

        Returns:
            tuple: (next_state, reward, done, info)
                next_state (numpy.ndarray): The next state vector.
                reward (float): The reward received.
                done (bool): Whether the episode is done.
                info (dict): Additional information (empty dict in this implementation).
        """
        # Convert action to ActionTuple
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.array([[action]]))

        # Take action in the environment
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        # Get the result of the action
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        # Check if the episode is done
        done = len(terminal_steps) > 0

        if done:
            # Use terminal_steps if the episode is done
            reward = terminal_steps.reward[0]
            next_state = terminal_steps.obs[0][0]
        else:
            # Use decision_steps if the episode is not done
            reward = decision_steps.reward[0]
            next_state = decision_steps.obs[0][0]

        return next_state, reward, done, {}

    def close(self):
        """
        Close the environment.
        """
        self.env.close()

    def render(self):
        """
        Render the environment (no-op in this implementation as Unity handles rendering).
        """
        pass  # Unity environment handles rendering

    def get_action_meanings(self):
        """
        Get the meanings of the actions.

        Returns:
            list: The meanings of the actions.
        """
        return ["FORWARD", "BACKWARD", "LEFT", "RIGHT"]
