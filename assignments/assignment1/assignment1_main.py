#! /usr/bin/env python3
"""
Author: Supreeth Suresh
Title: Assignment 1 - AI for multiagent systems

Instructions to run the code:
usage: python assignment1_main.py [-h] [--task {policy_iteration,value_iteration}] \\
    [--gamma GAMMA] [--epsilon EPSILON] [--max_iterations MAX_ITERATIONS] \\
    [--grid_size GRID_SIZE] [--stepReward STEPREWARD] [--goalReward GOALREWARD] \\
    [--valueFunctionInit {V,Q}] [--randomValueFunctionInit] [--randomPolicyInit] \\
    [--problem {1,2,3,4}] [--plotTable] [--goalStates GOALSTATES]

    options:
    -h, --help            show this help message and exit
    --task {policy_iteration,value_iteration}
    --gamma GAMMA                           Gamma for the value iteration
    --epsilon EPSILON                       Epsilon for the value iteration
    --max_iterations MAX_ITERATIONS         Maximum number of iterations for the value iteration and policy iteration
    --grid_size GRID_SIZE                   Size of the grid N
    --stepReward STEPREWARD                 Step reward
    --goalReward GOALREWARD                 Goal reward
    --valueFunctionInit {V,Q}               Type of value function used V or Q
    --randomValueFunctionInit               Initialize the value function with random values
    --randomPolicyInit                      Initialize the policy with random values
    --problem {1,2,3,4}                     Problem number
    --plotTable                             Plot the value function and policy
    --goalStates GOALSTATES                   Goal states list. Format list of tuples [(x, y), (x, y), ...]

Examples:
python assignment1_main.py \\
    --task policy_iteration \\
    --gamma 0.9 \\
    --epsilon 1e-6 \\
    --max_iterations 150 \\
    --grid_size 4 \\
    --stepReward -1 \\
    --goalReward 0 \\
    --valueFunctionInit V \\
    --randomValueFunctionInit False \\
    --randomPolicyInit False \\
    --plotTable True \\
    --goalStates [(0, 0), (3, 3)]
"""

import sys
import os
import argparse
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import math

random.seed(42)
np.random.seed(42)

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
        self.terminal_states = config.goalStates

        # Policy
        self.current_policy = {state: {action: 0 for action in self.actions} for state in self.states}
        self.new_policy = {state: {action: 0 for action in self.actions} for state in self.states}

        # Model hyperparameters
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.max_iterations = config.max_iterations
        
    ##### Initialize Value Function #####
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
    
    ##### Initialize Policy #####
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

    ##### Get Next States #####
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

    ##### Step #####
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

    ##### Policy Evaluation #####
    def calculateValueFunction(self):
        for state in self.states:
            # Check if the state is a terminal state, if so set the value to 0 and continue
            if state in self.terminal_states:
                self.newV[state] = 0
                continue
            
            # Reset newV for this state to 0
            self.newV[state] = 0
            
            # Calculate expected value for this state with current policy
            for action in self.actions:
                _, _, nextState, reward, done = self.step(state, action)
                self.newV[state] += self.current_policy[state][action] * (reward + self.gamma * self.currentV[nextState])
            
        return self.newV

    ##### Policy Improvement #####
    def policyImprovement(self):
        # Policy improvement phase
        for state in self.states:
            if state in self.terminal_states:
                continue
            n = self.getNextStates(state)
            for action in self.actions:
                self.current_policy[state][action] = 0 if state in self.terminal_states else self.current_policy[state][action]

            nextStatesVvalues = [self.currentV[n[action]] for action in self.actions]
            
            # Find the maximum value from the next states
            max_value = max(nextStatesVvalues)
            
            best_actions = []
            for i, action in enumerate(self.actions):
                if nextStatesVvalues[i] == max_value:
                    if n[action] in self.terminal_states:
                        best_actions.insert(0, action)
                    else:
                        best_actions.append(action)
            
            max_action = best_actions[0]
            
            for action in self.actions:
                self.current_policy[state][action] = 0 if action != max_action else 1

        return self.current_policy

    ##### Policy Iteration #####
    def policyIterationV(self, config):
        # Initialize the value function and policy
        self.initializeValueFunction(config)
        self.initializePolicy(config)
        self.newV = self.currentV.copy()
        converged = False
        
        logger.info(f"Starting policy iteration with {config.max_iterations} max iterations")
        count = 0
        VList = []
        while not converged:
            #logger.info(f"Iteration {i+1}")
            #### Policy evaluation ####
            for iter in range(config.max_iterations):
                oldV = self.currentV.copy()
                newV = self.calculateValueFunction()
                
                # Check convergence of value function
                currentVnp = np.array(list(self.currentV.values()))
                newVnp = np.array(list(self.newV.values()))
                if np.allclose(currentVnp, newVnp, atol=0.0001):
                    logger.info(f"Value function converged after {iter+1} evaluation iterations")
                    break
                    
                self.currentV = self.newV.copy()

            # Plot the value function after policy evaluation
            if config.plotTable:
                VList.append(self.currentV)
            
            #### Policy improvement ####
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
                logger.info(f"Policy converged after {count+1} iterations")
                converged = True
                break
            count += 1

        if not converged:
            logger.info(f"Policy iteration did not converge after {config.max_iterations} iterations")
        else:
            logger.info(f"Policy iteration converged successfully")
        
        # Plot the value function and policy
        if config.plotTable:
            self.plotValueFunction(self.currentV, "Final Value Function")
            self.plotOptimalPolicy(self.currentV, self.current_policy, "Final Optimal Policy")

        return self.current_policy

    ##### Get Output Deterministic Policy #####
    def getOutputDeterministicPolicy(self):
        # get next states for each state
        nextStates = {state: self.getNextStates(state) for state in self.states}
        policy = {state: {action: 0 for action in self.actions} for state in self.states}
        for state in self.states:
            if state in self.terminal_states:
                continue
            max_action = max(nextStates[state].items(), key=lambda x: self.currentV[x[1]])[0]
            policy[state] = max_action
        return policy

    ##### Value Iteration #####
    def valueIterationV(self, config):
        # Initialize the value function
        self.initializeValueFunction(config)
        converged = False
        count = 0
        VList = []
        while not converged:
            delta = 0
            for state in self.states:
                v = self.currentV[state]
                nextStates = self.getNextStates(state)

                for action in self.actions:
                    _, _, nextState, reward, done = self.step(state, action)
                    # uniform distribution for the next state
                    nextStateProb = 1/len(nextStates)

                    max_action = max(nextStates, key=lambda x: self.currentV[x])
                    nextState = nextStates[max_action]
                    v += nextStateProb * (reward + self.gamma * self.currentV[nextState])
                self.newV[state] = v
                delta = max(delta, math.fabs(v - self.currentV[state]))
            self.currentV = self.newV.copy()
            if delta < config.epsilon:
                logger.info(f"Value iteration converged after {count+1} iterations")
                converged = True
                break
            count += 1
        
        optimal_policy = self.getOutputDeterministicPolicy()

        if not converged:
            logger.info(f"Value iteration did not converge after {config.max_iterations} iterations")
        else:
            logger.info(f"Value iteration converged successfully")

        # Plot the value function and policy
        if config.plotTable:
            self.plotValueFunction(self.currentV, "Final Value Function")
            self.plotOptimalPolicy(self.currentV, optimal_policy, "Final Optimal Policy")

        return optimal_policy
            
        # # update V(s) = maximum action( sigma all next states (p(s'|s,a) * (r + gamma * V(s'))))
        # # calculate delta = max(abs(V(s) - V'(s))) for all s in stat
        # # if delta < epsilon, then stop


        # # construct the policy based on the value function
        # # pi(s) = argmax( sigma all next states (p(s'|s,a) * (r + gamma * V(s'))))
        # pass

    # Plot code is generated by VSCode - GitHub Copilot
    def plotOptimalPolicy(self, valueFunction: dict, policy: dict, title: str = "Optimal Policy"):
        logger.info(f"Plotting optimal policy with values and actions")
        
        # Create a grid to store the display text for each cell
        display_grid = []
        for row in range(self.grid_size):
            display_row = []
            for col in range(self.grid_size):
                state = (row, col)
                
                # Get the value for this state
                value = valueFunction.get(state, 0.0)
                
                # Get the optimal action for this state
                if state in self.terminal_states:
                    action_text = "TERM"
                else:
                    # Find the action with probability 1 (optimal action)
                    optimal_action = None
                    for action, prob in policy[state].items():
                        if prob == 1.0:
                            optimal_action = action
                            break
                    
                    if optimal_action:
                        # Convert action to arrow symbols
                        action_symbols = {
                            'up': '↑',
                            'down': '↓', 
                            'left': '←',
                            'right': '→'
                        }
                        action_text = action_symbols.get(optimal_action, optimal_action)
                    else:
                        action_text = "?"
                
                # Combine value and action in the cell
                cell_text = f"{value:.2f}\n{action_text}"
                display_row.append(cell_text)
            display_grid.append(display_row)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create table
        table = ax.table(cellText=display_grid, 
                        cellLoc='center', 
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Color code terminal states
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = (row, col)
                if state in self.terminal_states:
                    table[(row, col)].set_facecolor('#90EE90')
                else:
                    table[(row, col)].set_facecolor('#F0F0F0')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#90EE90', label='Terminal States'),
            plt.Rectangle((0,0),1,1, facecolor='#F0F0F0', label='Regular States')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        plt.show()
        plt.close()

    # Plot code is generated by VSCode - GitHub Copilot
    def plotValueFunction(self, valueFunction: dict, title: str = "Value Function"):
        logger.info(f"Plotting value function")
        
        # Create a grid to store the display text for each cell
        display_grid = []
        for row in range(self.grid_size):
            display_row = []
            for col in range(self.grid_size):
                state = (row, col)
                
                # Get the value for this state
                value = valueFunction.get(state, 0.0)
                
                # Format the value with appropriate precision
                if abs(value) < 0.01:
                    cell_text = f"{value:.3f}"
                else:
                    cell_text = f"{value:.2f}"
                
                display_row.append(cell_text)
            display_grid.append(display_row)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create table
        table = ax.table(cellText=display_grid, 
                        cellLoc='center', 
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2)
        
        # Color code terminal states and use color gradient for values
        max_value = max(valueFunction.values()) if valueFunction.values() else 0
        min_value = min(valueFunction.values()) if valueFunction.values() else 0
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = (row, col)
                value = valueFunction.get(state, 0.0)
                
                if state in self.terminal_states:
                    table[(row, col)].set_facecolor('#90EE90')  # Light green for terminal states
                else:
                    # Color gradient based on value (darker for lower values)
                    if max_value != min_value:
                        normalized_value = (value - min_value) / (max_value - min_value)
                        # Use a color gradient from light blue (high values) to light red (low values)
                        color_intensity = 0.3 + 0.7 * normalized_value  # Range from 0.3 to 1.0
                        if normalized_value > 0.5:
                            # Light blue to white for positive/high values
                            color = (0.7, 0.9, 1.0, color_intensity)
                        else:
                            # Light red to white for negative/low values
                            color = (1.0, 0.7, 0.7, color_intensity)
                        table[(row, col)].set_facecolor(color)
                    else:
                        table[(row, col)].set_facecolor('#F0F0F0')  # Light gray if all values are the same
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#90EE90', label='Terminal States'),
            plt.Rectangle((0,0),1,1, facecolor='#F0F0F0', label='Regular States')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        plt.show()
        plt.close()

    # def policyIterationQ(self, config):
    #     pass

    # def valueIterationQ(self, config):
    #     pass

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
    parser.add_argument("--randomValueFunctionInit", action="store_true", help="Randomly initialize the value function")
    parser.add_argument("--randomPolicyInit", action="store_false", help="Randomly initialize the policy")
    parser.add_argument("--problem", type=int, required=False, choices=[1, 2, 3 ,4], help="Problem number")
    parser.add_argument("--plotTable", action="store_true", help="Plot the value function and policy")
    parser.add_argument("--goalStates", type=tuple, default=[(0, 0), (3, 3)], help="Goal states list. Format list of tuples [(x, y), (x, y), ...]")
    config = parser.parse_args()

    if config.problem == 1:
        # Prob 1
        logger.info(f"Performing calculations for Prob 1: {config.task} with {config.valueFunctionInit} value function  \
                        and uniform distribution for policy initialization")
        config.stepReward = -1
        config.goalReward = 0
        config.gamma = 0.9
        config.epsilon = 1e-6
        config.max_iterations = 150
        config.grid_size = 4
        config.valueFunctionInit = "V"
        config.randomValueFunctionInit = False
        config.randomPolicyInit = False
        config.task = "policy_iteration"
        config.plotTable = True
        config.goalStates = [(0, 0), (3, 3)]
    elif config.problem == 2:
        # Prob 2
        logger.info(f"Performing calculations for Prob 2: {config.task} with {config.valueFunctionInit} value function  \
                        and uniform distribution for policy initialization")
        config.stepReward = -4
        config.goalReward = -1
        config.gamma = 0.9
        config.epsilon = 1e-6
        config.max_iterations = 150
        config.grid_size = 4
        config.valueFunctionInit = "V"
        config.randomValueFunctionInit = False
        config.randomPolicyInit = False
        config.task = "policy_iteration"
        config.plotTable = True
        config.goalStates = [(2, 2)]
    elif config.problem == 3:
        # Prob 3
        logger.info(f"Performing calculations for Prob 3: {config.task} with {config.valueFunctionInit} value function  \
                        and uniform distribution for policy initialization")
        config.stepReward = -1
        config.goalReward = 0
        config.gamma = 0.9
        config.epsilon = 1e-6
        config.max_iterations = 150
        config.grid_size = 4
        config.valueFunctionInit = "V"
        config.randomValueFunctionInit = False
        config.randomPolicyInit = False
        config.task = "policy_iteration"
        config.plotTable = True
        config.goalStates = [(2, 2)]
    elif config.problem == 4:
        # Prob 4
        logger.info(f"Performing calculations for Prob 4: {config.task} with {config.valueFunctionInit} value function  \
                        and uniform distribution for policy initialization")
        config.stepReward = -1
        config.goalReward = 0
        config.gamma = 0.9
        config.epsilon = 1e-6
        config.max_iterations = 150
        config.grid_size = 4
        config.valueFunctionInit = "V"
        config.randomValueFunctionInit = False
        config.randomPolicyInit = False
        config.task = "value_iteration"
        config.plotTable = True
        config.goalStates = [(2, 2)]

    # initialize the grid world
    grid_world = GridWorld(config)

    if config.valueFunctionInit == "V":
        if config.task == "policy_iteration":
            optimal_policy = grid_world.policyIterationV(config)
            #logger.info(f"Optimal policy: {optimal_policy}")
        elif config.task == "value_iteration":
            grid_world.valueIterationV(config)

    # elif config.valueFunctionInit == "Q":
    #     if config.task == "policy_iteration":
    #         optimal_policy = grid_world.policyIterationQ(config)
    #         logger.info(f"Optimal policy: {optimal_policy}")
    #     elif config.task == "value_iteration":
    #         grid_world.valueIterationQ(config)

if __name__ == "__main__":
    main()