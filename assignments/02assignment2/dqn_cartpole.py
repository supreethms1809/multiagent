"""
Assignment 2: Problem 1: Deep Q-learning on a cartpole balance problem.

Usage:
python dqn_cartpole.py 
    --env_name CartPole-v1       # Environment name
    --render                     # Render the environment
    --render_mode human          # Render mode
    --render_time 10             # Render time
    --max_steps 100              # Maximum steps per episode
    --batch_size 128             # Batch size
    --gamma 0.99                 # Discount factor
    --epsilon_start 0.9          # Epsilon start
    --epsilon_end 0.01           # Epsilon end
    --epsilon_decay 2500         # Epsilon decay
    --tau 0.005                  # Tau
    --lr 1e-4                    # Learning rate
    --memory_capacity 1000000    # Memory capacity
    --max_episodes 1000          # Maximum episodes
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import logging
import sys
import os
import random
import math
import argparse
import torch
import torch.nn as nn
from collections import namedtuple, deque

# Set the random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Select the device
device = torch.device("mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

# Initialize the environment
def initialize_environment(env_name, render_mode="human"):
    env = gym.make(env_name, render_mode=render_mode)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    return env, state_space, action_space

# Render the environment to show the cartpole balance animation
def render_environment(env, render_time):
    for i in range(render_time):
        env.reset(seed=42)
        env.render()
        time.sleep(0.1)

# Transition tuple for experience replay
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

# Replay buffer for experience replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Deep Q-network model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Deep Q-learning agent
class DQN_agent:
    def __init__(self, n_observations, n_actions, training_config, env, max_steps):
        # Environment
        self.env = env
        self.max_steps = max_steps
        self.n_observations = n_observations
        self.n_actions = n_actions

        # Logger
        self.logger = training_config["logger"]

        # Policy and target networks
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Training configuration
        self.batch_size = training_config["BATCH_SIZE"]
        self.gamma = training_config["GAMMA"]
        self.epsilon_start = training_config["EPSILON_START"]
        self.epsilon_end = training_config["EPSILON_END"]
        self.epsilon_decay = training_config["EPSILON_DECAY"]
        self.tau = training_config["TAU"]
        self.target_update_freq = training_config["TARGET_UPDATE_FREQ"]
        self.lr = training_config["LR"]
        self.memory_capacity = training_config["MEMORY_CAPACITY"]
        self.max_episodes = training_config["MAX_EPISODES"]

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

        # Memory
        self.steps_done = 0
        self.memory = ReplayMemory(self.memory_capacity)

    def select_action(self, state):
        # Select a random number between 0 and 1
        sample = random.random()
        # Calculate the epsilon threshold
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        # If the sample is greater than the epsilon threshold, select the action from Q network
        if sample > eps_threshold:
            # Select the action from Q network
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Select a random action based on epsilon greedy policy
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    # Train the DQN
    def train_dqn(self):
        # If the memory is less than the batch size, return
        if len(self.memory) < self.batch_size:
            return
        # Sample a batch of transitions from the memory
        transitions = self.memory.sample(self.batch_size)
        # Create a batch of transitions
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

        # Concatenate the non-final next states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute the state-action values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-100, 100)
        self.optimizer.step()
        
        return loss.item()

    def train_wrapper(self):
        total_reward_per_episode = []
        total_episode_duration = []
        
        for episode in range(self.max_episodes):
            # Initialize the environment and get its initial state for the episode
            state, info = self.env.reset()

            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0
            episode_duration = 0
            # Train the DQN for the episode
            for t in range(self.max_steps):
                # Select and perform an action using the policy network
                action = self.select_action(state)
                # Take the action and get the next state, reward, terminated, truncated, and info
                observation, reward, terminated, truncated, info = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                # If the episode is done, set the next state to None
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Push the transition to the memory
                self.memory.push(state, action, reward, next_state, done)

                # Update the state
                state = next_state

                # Increment step counter for epsilon decay
                self.steps_done += 1

                # Train the DQN
                self.train_dqn()

                # Update the target network every target_update_freq steps - still using soft update
                if self.steps_done % self.target_update_freq == 0:
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                    self.target_net.load_state_dict(target_net_state_dict)
                    # # Log every target update
                    # if self.steps_done % (self.target_update_freq * 10) == 0:
                    #     self.logger.info(f"Target network updated at step {self.steps_done}")

                # Update the episode reward and duration for plotting
                episode_reward += reward.item()
                episode_duration += 1
                if done:
                    total_episode_duration.append(episode_duration)
                    total_reward_per_episode.append(episode_reward)
                    if episode % 100 == 0:
                        self.logger.info(f"Episode {episode} reward: {total_reward_per_episode[-1]}, duration: {total_episode_duration[-1]}")
                    break

        return total_reward_per_episode, total_episode_duration

    # Save the model
    def save_model(self, episode):
        torch.save(self.policy_net.state_dict(), f"dqn_cartpole_model_{episode}.pth")
        self.logger.info(f"Model saved to dqn_cartpole_model_{episode}.pth")

    def plot_episode_rewards(self, total_episode_duration, problem):

        #smooth the total_episode_duration
        smooth_total_episode_duration = []
        window_size = 100
        for i in range(len(total_episode_duration)):
            start_idx = max(0, i - window_size + 1)
            smooth_total_episode_duration.append(np.mean(total_episode_duration[start_idx:i+1]))

        episodes = range(len(total_episode_duration))

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(episodes, total_episode_duration,'b-', alpha=0.9, linewidth=0.5, label='Episode Duration (raw)')
        ax.plot(episodes, smooth_total_episode_duration,'b-', linewidth=2, label='Episode Duration (smoothed)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Duration (Steps)')
        ax.set_title('Episode Duration Over Time')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'images/episode_rewards_{problem}.png', dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Episode Duration Over Time plot saved to images/episode_rewards_{problem}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default='1a', choices=['1a', '1b', '1c', '1d', '1e'], help="Problem number")
    parser.add_argument("--env_name", type=str, default='CartPole-v1', help="Environment name")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode")
    parser.add_argument("--render_time", type=int, default=10, help="Render time")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=0.9, help="Epsilon start")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Epsilon end")
    parser.add_argument("--epsilon_decay", type=int, default=2500, help="Epsilon decay")
    parser.add_argument("--tau", type=float, default=0.005, help="Tau")
    parser.add_argument("--target_update_freq", type=int, default=1, help="Target network update frequency (steps)")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--memory_capacity", type=int, default=10000, help="Memory capacity")
    parser.add_argument("--max_episodes", type=int, default=50, help="Maximum episodes")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    env_name = args.env_name
    render = args.render
    render_mode = args.render_mode
    max_steps = args.max_steps
    batch_size = args.batch_size
    # Gamma is discount factor
    gamma = args.gamma
    epsilon_start = args.epsilon_start
    epsilon_end = args.epsilon_end
    epsilon_decay = args.epsilon_decay
    tau = args.tau
    target_update_freq = args.target_update_freq
    # Learning rate
    lr = args.lr
    memory_capacity = args.memory_capacity
    # Maximum episodes
    max_episodes = args.max_episodes

    # Define the assignment problem configurations
    if args.problem == '1a':
        max_episodes = 50
        batch_size = 128
        gamma = 0.99
        epsilon_start = 0.9
        epsilon_end = 0.01
        epsilon_decay = 2500
        tau = 0.005
        target_update_freq = 1
        lr =3e-4
        memory_capacity = 10000
    elif args.problem == '1b':
        # Change the episode number from 50 to 1000
        max_episodes = 1000
        batch_size = 128
        gamma = 0.99
        epsilon_start = 0.9
        epsilon_end = 0.01
        epsilon_decay = 2500
        tau = 0.005
        target_update_freq = 1
        lr = 3e-4
        memory_capacity = 10000
    elif args.problem == '1c':
        max_episodes = 1000
        batch_size = 128
        # Change the gamma from 0.99 to 0.89
        gamma = 0.89
        epsilon_start = 0.9
        epsilon_end = 0.01
        epsilon_decay = 2500
        tau = 0.005
        target_update_freq = 1
        lr = 3e-4
        memory_capacity = 10000
    elif args.problem == '1d':
        max_episodes = 1000
        # Change the mini-batch size from 128 to 1500
        batch_size = 1500
        gamma = 0.99
        epsilon_start = 0.9
        epsilon_end = 0.01
        epsilon_decay = 2500
        tau = 0.005
        target_update_freq = 1
        lr = 3e-4
        memory_capacity = 10000
    elif args.problem == '1e':
        max_episodes = 1000
        batch_size = 128
        gamma = 0.99
        epsilon_start = 0.9     
        epsilon_end = 0.01
        epsilon_decay = 2500
        tau = 0.005
        target_update_freq = 1
        # Change the learning rate from 1e-4 to 1e-2
        lr = 1e-2
        memory_capacity = 10000
    else:
        raise ValueError(f"Unknown problem: {args.problem}. Must be one of: 1a, 1b, 1c, 1d, 1e")

    # Initialize environment
    env, state_space, action_space = initialize_environment(env_name=env_name, render_mode=render_mode)
    print(f"State space: {state_space}, Action space: {action_space}")
    n_observations = state_space
    n_actions = action_space

    # Training
    training_config = {
                    "BATCH_SIZE": batch_size,
                    "GAMMA": gamma,
                    "EPSILON_START": epsilon_start,
                    "EPSILON_END": epsilon_end,
                    "EPSILON_DECAY": epsilon_decay,
                    "TAU": tau,
                    "TARGET_UPDATE_FREQ": target_update_freq,
                    "LR": lr,
                    "MEMORY_CAPACITY": memory_capacity,
                    "MAX_EPISODES": max_episodes,
                    "logger": logger
            }

    logger.info(f"Training configuration: {training_config}")

    # Initialize Deep Q-learning agent
    agent = DQN_agent(n_observations, n_actions, training_config, env, max_steps)

    # Train the agent
    total_reward_per_episode, total_episode_duration = agent.train_wrapper()
    agent.plot_episode_rewards(total_episode_duration, args.problem)

    # Save the model
    # agent.save_model(max_episodes)

if __name__ == "__main__":
    main()