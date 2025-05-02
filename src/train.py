#!/usr/bin/env python3
# Navigation RL Agent - Training Script
# Author: Kashad J. Turner‑Warren

import argparse
import logging
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time

from src.environment_wrapper import BananaEnv
from src.dqn_agent import DQNAgent

# Configure logging
logger = logging.getLogger("navigation-rl-agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train or evaluate a DQN agent for navigation')
    parser.add_argument('--env-path', type=str, required=True, 
                        help='Path to the Unity environment executable')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Train the agent')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Evaluate the agent')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes to train for')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to load model weights from')
    parser.add_argument('--save-path', type=str, default='models/checkpoint_final.pt',
                        help='Path to save model weights to')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--worker-id', type=int, default=0,
                        help='Worker ID for Unity environment')
    parser.add_argument('--no-graphics', action='store_true', default=False,
                        help='Run Unity environment without graphics')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Use GPU for training if available')

    return parser.parse_args()

def train_dqn(env, args):
    """Train a DQN agent in the Unity environment."""
    logger.info("Starting training with the following parameters:")
    logger.info(f"  Episodes: {args.episodes}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Gamma: {args.gamma}")
    logger.info(f"  Save path: {args.save_path}")

    # Create the agent
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        seed=args.seed,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr
    )

    # Load model if specified
    if args.load_path is not None:
        logger.info(f"Loading model from {args.load_path}")
        agent.load(args.load_path)

    # Training loop
    scores = []
    scores_window = deque(maxlen=100)
    solved = False
    solved_episode = 0

    for i_episode in range(1, args.episodes + 1):
        state = env.reset()
        score = 0

        while True:
            # Select and take action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Update agent
            agent.step(state, action, reward, next_state, done)

            # Update state and score
            state = next_state
            score += reward

            if done:
                break

        # Save score and print progress
        scores.append(score)
        scores_window.append(score)
        mean_score = np.mean(scores_window)

        logger.info(f"Episode {i_episode}/{args.episodes} | Score: {score:.2f} | Average: {mean_score:.2f}")

        # Check if environment is solved
        if mean_score >= 13.0 and not solved:
            solved = True
            solved_episode = i_episode
            logger.info(f"Environment solved in {solved_episode} episodes! Average score: {mean_score:.2f}")

            # Save the solved episode number
            os.makedirs('results', exist_ok=True)
            with open('results/solved_at_episode.txt', 'w') as f:
                f.write(f"The agent solved the environment (achieved an average score of +13 over 100 consecutive episodes) at episode {solved_episode}.")

        # Save model periodically
        if i_episode % 100 == 0:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            checkpoint_path = f"models/checkpoint_{i_episode}.pt"
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    agent.save(args.save_path)
    logger.info(f"Training complete. Final model saved to {args.save_path}")

    # Plot scores
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title('DQN Training Scores')

    # Add a horizontal line at 13.0
    plt.axhline(y=13.0, color='r', linestyle='-', label='Solved Threshold')

    # Add a vertical line at the solved episode
    if solved:
        plt.axvline(x=solved_episode, color='g', linestyle='--', label=f'Solved at Episode {solved_episode}')

    plt.legend()

    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/rewards_plot.png')
    logger.info("Score plot saved to results/rewards_plot.png")

def evaluate_dqn(env, args):
    """Evaluate a trained DQN agent in the Unity environment."""
    if args.load_path is None:
        logger.error("No model path specified for evaluation. Use --load-path.")
        return

    if not os.path.exists(args.load_path):
        logger.error(f"Model file not found: {args.load_path}")
        return

    logger.info(f"Evaluating agent using model from {args.load_path}")

    # Create the agent
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        seed=args.seed
    )

    # Load the model
    agent.load(args.load_path)

    # Run evaluation episodes
    num_episodes = 10
    scores = []

    for i in range(num_episodes):
        state = env.reset()
        score = 0

        while True:
            # Select action using the trained policy (no exploration)
            action = agent.act(state, eps=0.0)

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)

            # Update score and state
            score += reward
            state = next_state

            if done:
                break

        scores.append(score)
        logger.info(f"Episode {i+1}/{num_episodes}, Score: {score}")

    logger.info(f"Evaluation complete. Average score: {np.mean(scores):.2f}")

def main():
    """Main function to train or evaluate a DQN agent."""
    args = parse_args()

    if not args.train and not args.eval:
        logger.error("Please specify either --train or --eval")
        return

    # Set device for PyTorch
    if args.gpu and torch.cuda.is_available():
        logger.info("Using GPU for training")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        logger.info("Using CPU for training")
        torch.set_default_tensor_type('torch.FloatTensor')

    # Create the environment
    logger.info(f"Creating Unity environment from {args.env_path}")
    env = BananaEnv(
        env_path=args.env_path,
        worker_id=args.worker_id,
        seed=args.seed,
        no_graphics=args.no_graphics,
        train_mode=args.train
    )

    try:
        if args.train:
            train_dqn(env, args)

        if args.eval:
            evaluate_dqn(env, args)

    finally:
        # Close the environment
        env.close()
        logger.info("Environment closed")

if __name__ == "__main__":
    main()
