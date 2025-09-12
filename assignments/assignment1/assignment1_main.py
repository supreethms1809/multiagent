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

        # Grid
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # States and actions
        self.states = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.actions = ["up", "down", "left", "right"]
        self.num_actions = len(self.actions)

        # Rewards
        self.reward = config.stepReward
        self.goal_reward = config.goalReward

        # Value function
        self.currentV = {state: 0 for state in self.states}
        self.currentQ = {state: {action: 0 for action in self.actions} for state in self.states}
        self.newV = {state: 0 for state in self.states}
        self.newQ = {state: {action: 0 for action in self.actions} for state in self.states}

        # Terminal states
        self.terminal_states = [(0, 0), (self.grid_size-1, self.grid_size-1)]

        # Policy
        self.current_policy = {state: {action: 0 for action in self.actions} for state in self.states}
        self.new_policy = {state: {action: 0 for action in self.actions} for state in self.states}

        # Model hyperparameters
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.max_iterations = config.max_iterations
        
    
    def initializeValueFunction(self, config):
        if config.randomValueFunctionInit:
            logger.info("Using V value function with random initialization")
            if config.valueFunctionInit == "V":
                self.currentV = {state: np.random.rand() for state in self.states}
            elif config.valueFunctionInit == "Q":
                logger.info("Using Q value function with random initialization")
                self.currentQ = {state: {action: np.random.rand() for action in self.actions} for state in self.states}
        else:
            if config.valueFunctionInit == "V":
                logger.info("Using V value function and initializing to 0")
                self.currentV = {state: 0 for state in self.states}
            elif config.valueFunctionInit == "Q":
                logger.info("Using Q value function and initializing to 0")
                self.currentQ = {state: {action: 0 for action in self.actions} for state in self.states}
        return self.currentV, self.currentQ
    
    def initializePolicy(self, config):
        if config.randomPolicyInit:
            logger.info("Using random policy initialization")
            self.current_policy = {state: {action: random.random() for action in self.actions} for state in self.states}
            for state in self.terminal_states:
                self.current_policy[state] = {action: 0 for action in self.actions}
        else:
            logger.info("Using uniform policy initialization")
            self.current_policy = {state: {action: 1/self.num_actions for action in self.actions} for state in self.states}
            for state in self.terminal_states:
                self.current_policy[state] = {action: 0 for action in self.actions}
        return self.current_policy

    def getNextStates(self, current_state):
        x, y = current_state
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
                nextState = current_state
            nextStates[action] = nextState
        return nextStates

    def step(self, current_state, action):
        x, y = current_state
        if action == "up":
            nextState = (x-1, y)
        elif action == "down":
            nextState = (x+1, y)
        elif action == "left":
            nextState = (x, y-1)
        elif action == "right":
            nextState = (x, y+1)
        
        if nextState not in self.states:
            nextState = current_state

        if nextState in self.terminal_states:
            done = True
            reward = self.goal_reward
        else:
            done = False
            reward = self.reward
            
        return current_state, action, nextState, reward, done

    def calculateValueFunction(self):
        for state in self.states:
            # Check if the state is a terminal state, set the value to 0 and continue
            if state in self.terminal_states:
                self.newV[state] = 0
                continue
            
            # Reset newV for this state
            self.newV[state] = 0
            
            # Calculate expected value for this state under current policy
            for action in self.actions:
                _, _, nextState, reward, done = self.step(state, action)
                self.newV[state] += self.current_policy[state][action] * (reward + self.gamma * self.currentV[nextState])
                #logger.info(f"State: {state}, Action: {action}, NextState: {nextState}, Reward: {reward}, Policy: {self.current_policy[state][action]}, V: {self.newV[state]}") if state == (0, 1) else None
            
        return self.newV

    def policyImprovement(self):
        #logger.info(f"Current value function: {self.currentV}")
        for state in self.states:
            if state in self.terminal_states:
                continue
            n = self.getNextStates(state)
            for action in self.actions:
                self.current_policy[state][action] = 0 if state in self.terminal_states else self.current_policy[state][action]

            #logger.info(f"Current State: {state} and Next States: {n}")
            nextStatesVvalues = [self.currentV[n[action]] for action in self.actions]
            #logger.info(f"Next States V values: {nextStatesVvalues}")
            
            # Find the maximum value
            max_value = max(nextStatesVvalues)
            
            best_actions = []
            for i, action in enumerate(self.actions):
                if nextStatesVvalues[i] == max_value:
                    if n[action] in self.terminal_states:
                        best_actions.insert(0, action)
                    else:
                        best_actions.append(action)
            
            max_action = best_actions[0]
            
            #logger.info(f"Max action: {max_action}")
            for action in self.actions:
                self.current_policy[state][action] = 0 if action != max_action else 1

        return self.current_policy

    def policyIterationV(self, config):
        # Initialize the value function and policy
        self.initializeValueFunction(config)
        self.initializePolicy(config)
        self.newV = self.currentV.copy()
        converged = False
        
        logger.info(f"Starting policy iteration with {config.max_iterations} max iterations")
        
        for i in range(config.max_iterations):
            #logger.info(f"Iteration {i+1}")
            
            # Policy evaluation - iterate until value function converges
            for iter in range(config.max_iterations):
                oldV = self.currentV.copy()
                self.newV = self.calculateValueFunction()
                
                # Check convergence of value function
                currentVnp = np.array(list(self.currentV.values()))
                newVnp = np.array(list(self.newV.values()))
                if np.allclose(currentVnp, newVnp, atol=0.0001):
                    logger.info(f"Value function converged after {iter+1} evaluation iterations")
                    break
                    
                self.currentV = self.newV.copy()
            
            # Policy improvement
            old_policy = {state: policy.copy() for state, policy in self.current_policy.items()}
            self.new_policy = self.policyImprovement()
            
            # Check if policy has changed
            policy_changed = False
            for state in self.states:
                if state not in self.terminal_states:
                    for action in self.actions:
                        if abs(old_policy[state][action] - self.new_policy[state][action]) > 1e-6:
                            policy_changed = True
                            break
                    if policy_changed:
                        break
            
            self.current_policy = self.new_policy.copy()
            
            if not policy_changed:
                logger.info(f"Policy converged after {i+1} iterations")
                converged = True
                break

        if not converged:
            logger.info(f"Policy iteration did not converge after {config.max_iterations} iterations")
        else:
            logger.info(f"Policy iteration converged successfully")
            
        return self.current_policy

    def valueIterationV(self, config):
        pass

    def policyIterationQ(self, config):
        pass

    def valueIterationQ(self, config):
        pass

def main():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="policy_iteration", choices=["policy_iteration", "value_iteration"])
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma for the value iteration")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Epsilon for the value iteration")
    parser.add_argument("--max_iterations", type=int, default=200, help="Maximum number of iterations for the value iteration and policy iteration")
    parser.add_argument("--grid_size", type=int, default=4, help="Size of the grid N")
    parser.add_argument("--stepReward", type=int, default=-1, help="Step reward")
    parser.add_argument("--goalReward", type=int, default=0, help="Goal reward")
    parser.add_argument("--valueFunctionInit", type=str, default="V", choices=["V", "Q"], help="Type of value function used V or Q")
    parser.add_argument("--randomValueFunctionInit", type=bool, default=False, help="Randomly initialize the value function")
    parser.add_argument("--randomPolicyInit", type=bool, default=False, help="Randomly initialize the policy")
    config = parser.parse_args()
    
    # initialize the grid world
    grid_world = GridWorld(config)

    if config.valueFunctionInit == "V":
        if config.task == "policy_iteration":
            optimal_policy = grid_world.policyIterationV(config)
            logger.info(f"Optimal policy: {optimal_policy}")
        elif config.task == "value_iteration":
            grid_world.valueIterationV(config)
    elif config.valueFunctionInit == "Q":
        if config.task == "policy_iteration":
            optimal_policy = grid_world.policyIterationQ(config)
            logger.info(f"Optimal policy: {optimal_policy}")
        elif config.task == "value_iteration":
            grid_world.valueIterationQ(config)

if __name__ == "__main__":
    main()