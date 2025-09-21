import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GridWorld:
    def __init__(self, config):
        self.config = config
        self.grid_size = config.grid_size
        self.states = [(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])]
        self.goal_states = config.goal_states
        self.start_state = config.start_state
        self.cliff_states = config.cliff_states
        self.cliff_reward = config.cliff_reward
        self.step_reward = config.step_reward
        self.goal_reward = config.goal_reward
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.max_episodes = config.max_episodes
        self.max_steps_per_episode = config.max_steps_per_episode
        self.actions = ["up", "down", "left", "right"]
        self.qnetwork = Qnetwork(len(self.states), len(self.actions))
        self.optimizer = optim.AdamW(self.qnetwork.parameters(), lr=config.learning_rate, betas=config.betas)
        self.loss_fn = nn.MSELoss()

    def plot_gridworld(self):
        pass

class Qnetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(Qnetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=tuple, default=(6, 10), help="Size of the grid")
    parser.add_argument("--goal_states", type=list, default=[(5, 9)], help="Goal states")
    parser.add_argument("--start_state", type=list, default=[(0, 0)], help="Start state")
    parser.add_argument("--cliff_states", type=list, default=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 7), (2, 8)], help="Cliff states")
    parser.add_argument("--cliff_reward", type=int, default=-100, help="Reward for falling into a cliff")
    parser.add_argument("--step_reward", type=int, default=-4, help="Reward for each step")
    parser.add_argument("--goal_reward", type=int, default=100, help="Reward for reaching the goal")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon for epsilon-greedy policy")
    parser.add_argument("--max_episodes", type=int, default=10000, help="Maximum number of episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="Betas for AdamW optimizer")
    args = parser.parse_args()
    config = args
    gridworld = GridWorld(config)


if __name__ == "__main__":
    main()