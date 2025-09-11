import sys
import os
import argparse
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GridWorld:
    def __init__(self, grid_size: int, stepReward: int, goalReward: int):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.actions = ["up", "down", "left", "right"]
        self.num_actions = len(self.actions)
        self.reward = stepReward
        self.goal_reward = goalReward
        self.currentV = np.empty((grid_size, grid_size))
        self.currentQ = np.empty((grid_size, grid_size, self.num_actions))
        self.newV = np.empty((grid_size, grid_size))
        self.newQ = np.empty((grid_size, grid_size, self.num_actions))
        self.terminal_states = [[0, 0], [grid_size-1, grid_size-1]]
        self.policy = {action: 0 for action in self.actions}
        
    
    def initializeValueFunction(self, randomValueFunctionInit: bool) -> tuple[np.ndarray, np.ndarray]:
        if randomValueFunctionInit:
            logger.info("Initializing value function to random distribution")
            self.currentV = np.random.rand(self.grid_size, self.grid_size)
            self.currentQ = np.random.rand(self.grid_size, self.grid_size, self.num_actions)
        else:
            logger.info("Initializing value function to 0")
            self.currentV = np.zeros((self.grid_size, self.grid_size))
            self.currentQ = np.zeros((self.grid_size, self.grid_size, self.num_actions))
        return self.currentV, self.currentQ
    
    def initializePolicy(self, randomPolicyInit: bool) -> dict:
        if randomPolicyInit:
            logger.info("Initializing policy to random distribution")
            self.policy = {action: random.random() for action in self.actions}
        else:
            logger.info("Initializing policy to uniform distribution")
            self.policy = {action: 1/self.num_actions for action in self.actions}
        return self.policy

    def policyIteration(self, gamma: float, epsilon: float, max_iterations: int):
        # policy evaluation


        # policy improvement


        pass

    def valueIteration(self, gamma: float, epsilon: float, max_iterations: int):
        pass

def main():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="policy_iteration", choices=["policy_iteration", "value_iteration"])
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma for the value iteration")
    parser.add_argument("--epsilon", type=float, default=0.0001, help="Epsilon for the value iteration")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of iterations for the value iteration")
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid N")
    parser.add_argument("--stepReward", type=int, default=-1, help="Step reward")
    parser.add_argument("--goalReward", type=int, default=0, help="Goal reward")
    parser.add_argument("--randomValueFunctionInit", type=bool, default=False, help="Randomly initialize the value function")
    parser.add_argument("--randomPolicyInit", type=bool, default=False, help="Randomly initialize the policy")
    args = parser.parse_args()
    
    # initialize the grid world
    grid_world = GridWorld(args.grid_size, args.stepReward, args.goalReward)
    grid_world.initializeValueFunction(args.randomValueFunctionInit)
    logger.info(f"Current V: {grid_world.currentV}")
    logger.info(f"Current Q: {grid_world.currentQ}")
    logger.info("--------------------------------")
    policy = grid_world.initializePolicy(args.randomPolicyInit)
    logger.info(f"Policy: {policy}")
    logger.info("--------------------------------")

    if args.task == "policy_iteration":
        grid_world.policyIteration(args.gamma, args.epsilon, args.max_iterations)
    elif args.task == "value_iteration":
        grid_world.valueIteration(args.gamma, args.epsilon, args.max_iterations)

if __name__ == "__main__":
    main()