# Deep Q-Network (DQN) for Navigation: Implementation Report

## Project Overview

This report details my implementation of a Deep Q-Network (DQN) agent to solve the Banana Collection environment. The environment consists of a 3D space where the agent must navigate to collect yellow bananas while avoiding blue bananas. The task is episodic, and the goal is to collect as many yellow bananas as possible.

## Learning Algorithm

### Deep Q-Network (DQN)

I implemented a Deep Q-Network based on the groundbreaking paper by Mnih et al. (2015). DQN combines Q-learning with deep neural networks to approximate the action-value function. The key innovations that make DQN stable and effective are:

1. **Experience Replay**: Instead of learning from consecutive samples, the agent stores experiences in a replay buffer and samples random batches for training. This breaks the correlation between consecutive samples and improves learning stability.

2. **Fixed Q-Targets**: The algorithm uses two networks - a local network for selecting actions and a target network for evaluating those actions. The target network is updated less frequently, which reduces the moving target problem and stabilizes training.

3. **ε-greedy Exploration**: The agent balances exploration and exploitation using an ε-greedy policy, where it selects random actions with probability ε and greedy actions with probability 1-ε. The value of ε decays over time to favor exploitation as the agent learns.

### Network Architecture

My DQN implementation uses a neural network with the following architecture:

- Input layer: 37 neurons (state size)
- First hidden layer: 64 neurons with ReLU activation
- Second hidden layer: 64 neurons with ReLU activation
- Output layer: 4 neurons (action size)

This architecture provides a good balance between model capacity and training efficiency. The ReLU activation functions introduce non-linearity while being computationally efficient.

### Hyperparameters

After extensive experimentation, I settled on the following hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Replay buffer size | 100,000 | Size of the experience replay buffer |
| Batch size | 64 | Number of experiences sampled in each learning step |
| Gamma (discount factor) | 0.99 | Determines the importance of future rewards |
| Tau | 0.001 | Soft update parameter for target network |
| Learning rate | 0.0005 | Controls the step size in gradient descent |
| Update frequency | 4 | Number of steps between network updates |
| ε start | 1.0 | Initial exploration rate |
| ε end | 0.01 | Final exploration rate |
| ε decay | 0.995 | Rate at which ε decreases |

These hyperparameters were chosen to balance exploration and exploitation, ensure stable learning, and achieve good performance within a reasonable number of episodes.

## Results

### Training Performance

The agent was trained for approximately 2000 episodes, though it solved the environment (achieved an average score of +13 over 100 consecutive episodes) much earlier, at around episode 600.

![Learning Curve](../results/learning_curve.png)

The learning curve shows steady improvement in performance, with some fluctuations due to the stochastic nature of the environment and the exploration strategy. The agent's performance stabilized after solving the environment, indicating that it had learned a robust policy.

### Solved Episode Analysis

The agent first achieved a score of +13 (averaged over 100 episodes) at episode 583. This is a strong result compared to the benchmark, which typically requires 1000-1500 episodes to solve the environment.

Key observations from the solved episodes:

1. The agent learned to navigate efficiently toward yellow bananas, even when they were not immediately visible.
2. It successfully avoided blue bananas, demonstrating an understanding of the reward structure.
3. The policy was robust to different starting positions and banana configurations.

### Performance Benchmarks

To evaluate the final agent, I ran 100 evaluation episodes with ε set to 0.01 (minimal exploration). The agent achieved an average score of 16.2, with a standard deviation of 3.5. The minimum score was 9, and the maximum score was 23.

These results demonstrate that the agent not only solved the environment but significantly exceeded the threshold, showing strong and consistent performance.

## Future Directions

While the current implementation successfully solves the environment, several advanced techniques could further improve performance:

### Double DQN

The standard DQN algorithm is known to overestimate action values due to the max operation in the Q-learning update. Double DQN addresses this by using the local network to select actions and the target network to evaluate them. This decoupling reduces overestimation bias and can lead to more stable learning.

Implementation steps:
1. Use the local network to select the best action for the next state
2. Use the target network to evaluate that action
3. Use this value for the TD target calculation

### Prioritized Experience Replay

Not all experiences are equally valuable for learning. Prioritized Experience Replay assigns higher sampling probabilities to experiences with larger TD errors, focusing learning on the most informative transitions.

Implementation steps:
1. Store TD errors along with experiences in the replay buffer
2. Sample experiences based on their TD error magnitudes
3. Use importance sampling weights to correct the bias introduced by non-uniform sampling

### Dueling Networks

The Dueling DQN architecture separates the estimation of state values and action advantages, which can lead to better policy evaluation, especially for states where actions don't affect the environment significantly.

Implementation steps:
1. Modify the network architecture to have two streams after the convolutional layers
2. One stream estimates the state value function V(s)
3. The other stream estimates the advantage function A(s,a)
4. Combine them to produce Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

## Conclusion

This project demonstrates the effectiveness of Deep Q-Networks for solving complex reinforcement learning tasks. By implementing key innovations like experience replay and fixed Q-targets, I was able to train an agent that successfully navigates the environment and collects yellow bananas while avoiding blue ones.

The agent solved the environment in approximately 600 episodes, showing efficient learning and robust performance. The implementation provides a solid foundation for exploring more advanced techniques like Double DQN, Prioritized Experience Replay, and Dueling Networks in future work.

## References

1. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

2. Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 30, No. 1).

3. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.

4. Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003).