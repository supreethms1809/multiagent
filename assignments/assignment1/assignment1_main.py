import sys
import os
import argparse
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GridWorld:
    def __init__(self, config):
        self.grid_size = config.grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.states = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.actions = ["up", "down", "left", "right"]
        self.num_actions = len(self.actions)
        self.reward = config.stepReward
        self.goal_reward = config.goalReward
        self.currentV = np.empty((self.grid_size, self.grid_size))
        self.currentQ = np.empty((self.grid_size, self.grid_size, self.num_actions))
        self.newV = np.empty((self.grid_size, self.grid_size))
        self.newQ = np.empty((self.grid_size, self.grid_size, self.num_actions))
        self.terminal_states = [(0, 0), (self.grid_size-1, self.grid_size-1)]
        self.policy = {state: self.actions for state in self.states}
        self.new_policy = {state: self.actions for state in self.states}
        self.optimal_policy = {state: self.actions for state in self.states}
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.max_iterations = config.max_iterations
        
    
    def initializeValueFunction(self, config):
        if config.randomValueFunctionInit:
            logger.info("Using V value function with random initialization")
            if config.valueFunctionInit == "V":
                self.currentV = np.random.rand(self.grid_size, self.grid_size)
            elif config.valueFunctionInit == "Q":
                logger.info("Using Q value function with random initialization")
                self.currentQ = np.random.rand(self.grid_size, self.grid_size, self.num_actions)
        else:
            if config.valueFunctionInit == "V":
                logger.info("Using V value function and initializing to 0")
                self.currentV = np.zeros((self.grid_size, self.grid_size))
            elif config.valueFunctionInit == "Q":
                logger.info("Using Q value function and initializing to 0")
                self.currentQ = np.zeros((self.grid_size, self.grid_size, self.num_actions))
        return self.currentV, self.currentQ
    
    def initializePolicy(self, config):
        if config.randomPolicyInit:
            logger.info("Using random policy initialization")
            self.policy = {state: {action: random.random() for action in self.actions} for state in self.states}
            for state in self.terminal_states:
                self.policy[state] = {action: 0 for action in self.actions}
        else:
            logger.info("Using uniform policy initialization")
            self.policy = {state: {action: 1/self.num_actions for action in self.actions} for state in self.states}
            for state in self.terminal_states:
                self.policy[state] = {action: 0 for action in self.actions}
        return self.policy

    def getNextStates(self, state):
        x, y = state
        nextStates = {}
        for action in self.actions:
            if action == "up":
                nextState = (x-1, y)
            elif action == "down":
                nextState = (x+1, y)
            elif action == "left":
                nextState = (x, y-1)
            elif action == "right":
                nextState = (x, y+1)
            if nextState not in self.states:
                nextState = state
            nextStates[action] = nextState
        return nextStates

    def step(self, state, action):
        x, y = state
        if action == "up":
            nextState = (x-1, y)
        elif action == "down":
            nextState = (x+1, y)
        elif action == "left":
            nextState = (x, y-1)
        elif action == "right":
            nextState = (x, y+1)
        
        if nextState not in self.states:
            nextState = state

        if nextState in self.terminal_states:
            done = True
            reward = self.goal_reward
        else:
            done = False
            reward = self.reward
            
        return state, action, nextState, reward, done

    def calculateValueFunction(self):
        for state in self.states:
            if state in self.terminal_states:
                self.newV[state] = 0
                continue
            currentState = state
            V = 0
            for action in self.actions:
                state, action, nextState, reward, done = self.step(currentState, action)
                V += self.policy[currentState][action] * (reward + self.gamma * self.currentV[nextState])
                #logger.info(f"Current State: {currentState} and Next State: {nextState} and Reward: {reward} and Action: {action} and V: {V}") if state == (0, 1) else None
            self.newV[state] = V
            logger.info(f"Current V: {self.currentV[state]} and New V: {self.newV[state]}")
        return self.newV

    def policyImprovement(self):
        #logger.info(f"Current value function: {self.currentV}")
        for state in self.states:
            if state in self.terminal_states:
                continue
            n = self.getNextStates(state)
            for action in self.actions:
                self.policy[state][action] = 0 if state in self.terminal_states else self.policy[state][action]

            logger.info(f"Cuurent State: {state} and Value of n: {n}")
            nextStatesVvalues = [np.max(self.currentV[n[action]]) for action in self.actions]
            logger.info(f"Next States V values: {nextStatesVvalues}")
            max_action = self.actions[nextStatesVvalues.index(max(nextStatesVvalues))]
            logger.info(f"Max action: {max_action}")
            for action in self.actions:
                self.policy[state][action] = 0 if action != max_action else 1
            logger.info(f"Policy: {self.policy[state]}")
            logger.info(f"********************************************************")

        return self.policy

    def policyIteration(self, config):
        self.initializeValueFunction(config)
        self.initializePolicy(config)
        self.newV = self.currentV.copy()
        converged = False
        #logger.info(f"Initial policy: {self.policy}")
        # policy evaluation
        for i in range(config.max_iterations):
            # Calculate value function
            self.newV = self.calculateValueFunction()

            if np.allclose(self.currentV, self.newV, atol=0.0001):
                converged = True
                break

            self.currentV = self.newV.copy()

            # policy improvement
            self.new_policy = self.policyImprovement()
            self.policy = self.new_policy.copy()

        self.optimal_policy = self.new_policy
        if converged:
            logger.info(f"Converged ")
        return self.optimal_policy

    def valueIteration(self, config):
        pass

def main():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="policy_iteration", choices=["policy_iteration", "value_iteration"])
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma for the value iteration")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Epsilon for the value iteration")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum number of iterations for the value iteration")
    parser.add_argument("--grid_size", type=int, default=4, help="Size of the grid N")
    parser.add_argument("--stepReward", type=int, default=-1, help="Step reward")
    parser.add_argument("--goalReward", type=int, default=0, help="Goal reward")
    parser.add_argument("--valueFunctionInit", type=str, default="V", choices=["V", "Q"], help="Type of value function used V or Q")
    parser.add_argument("--randomValueFunctionInit", type=bool, default=False, help="Randomly initialize the value function")
    parser.add_argument("--randomPolicyInit", type=bool, default=False, help="Randomly initialize the policy")
    config = parser.parse_args()
    
    # initialize the grid world
    grid_world = GridWorld(config)

    if config.task == "policy_iteration":
        optimal_policy = grid_world.policyIteration(config)
        logger.info(f"Optimal policy: {optimal_policy}")
    elif config.task == "value_iteration":
        grid_world.valueIteration(config)

if __name__ == "__main__":
    main()