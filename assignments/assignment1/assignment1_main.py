#! /usr/bin/env python3
"""
Author: Supreeth Suresh
Title: Assignment 1 - AI for multiagent systems

Instructions to run the code:
usage: python assignment1_main.py [-h] [--task {policy_iteration,value_iteration}] \\
    [--gamma GAMMA] [--epsilon EPSILON] [--max_iterations MAX_ITERATIONS] \\
    [--grid_size grid_size] [--stepReward STEPREWARD] [--goalReward GOALREWARD] \\
    [--valueFunctionInit {V,Q}] [--randomValueFunctionInit] [--randomPolicyInit] \\
    [--problem {1,2,3,4}] [--plotTable] [--goalStates GOALSTATES]

    options:
    -h, --help            show this help message and exit
    --task {policy_iteration,value_iteration}
    --gamma GAMMA                           Gamma for the value iteration
    --epsilon EPSILON                       Epsilon for the value iteration
    --max_iterations MAX_ITERATIONS         Maximum number of iterations for the value iteration and policy iteration
    --grid_size grid_size                   Size of the grid N
    --stepReward STEPREWARD                 Step reward
    --goalReward GOALREWARD                 Goal reward
    --valueFunctionInit {V,Q}               Type of value function used V or Q
    --randomValueFunctionInit               Initialize the value function with random values
    --uniformPolicyInit                     Initialize the policy with uniform distribution
    --problem {1,2,3,4}                     Problem number
    --plotTable                             Plot the value function and policy
    --goalStates GOALSTATES                   Goal states list. Format list of tuples [(x, y), (x, y), ...]
    --splStates SPLSTATES                   SPL states list. Format list of tuples [(x, y), (x, y), ...]
    --splReward SPLREWARD                   Special state reward

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
    --randomValueFunctionInit \\
    --uniformPolicyInit \\
    --plotTable \\
    --goalStates [(0, 0), (3, 3)] \\
    --splStates [(2, 2)] \\
    --splReward -1
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
        self.spl_reward = config.splReward

        # Value function
        self.currentV = {state: 0 for state in self.states}
        self.currentQ = {state: {action: 0 for action in self.actions} for state in self.states}
        self.newV = {state: 0 for state in self.states}
        self.newQ = {state: {action: 0 for action in self.actions} for state in self.states}

        # Terminal states
        self.terminal_states = config.goalStates
        self.spl_states = config.splStates

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
                for state in self.terminal_states:
                    self.currentV[state] = 0
            elif config.valueFunctionInit == "Q":
                logger.info("Using Q value function with random initialization")
                self.currentQ = {state: {action: np.random.rand() for action in self.actions} for state in self.states}
                for state in self.terminal_states:
                    self.currentQ[state] = {action: 0 for action in self.actions}
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
        if not config.uniformPolicyInit:
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
        # Check if the next state is a special state (For problem 2, 3, 4)
        elif self.spl_reward is not None and self.spl_states is not None:
            if nextState in self.spl_states:
                done = False
                reward = self.spl_reward
            else:
                done = False
                reward = self.reward
        else:
            done = False
            reward = self.reward

        #logger.info(f"State: {current_state}, Action: {action}, Next State: {nextState}, Reward: {reward}, Done: {done}")
            
        return current_state, action, nextState, reward, done

    ##### Policy Evaluation #####
    def PolicyEvaluationcalculateValueFunction(self):
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
                for action in self.actions:
                    self.current_policy[state][action] = 0
                continue
                
            # Calculate V-value for each action and find the best one
            max_v_value = float('-inf')
            best_actions = []
            
            for action in self.actions:
                _, _, nextState, reward, _ = self.step(state, action)
                v_value = reward + self.gamma * self.currentV[nextState]
                
                if v_value > max_v_value:
                    max_v_value = v_value
                    best_actions = [action]
                elif v_value == max_v_value:
                    best_actions.append(action)
            
            # If multiple actions have the same V-value, prefer terminal states
            if len(best_actions) > 1:
                terminal_actions = [a for a in best_actions if self.getNextStates(state)[a] in self.terminal_states]
                if terminal_actions:
                    best_actions = terminal_actions
            
            max_action = best_actions[0]
            
            for action in self.actions:
                self.current_policy[state][action] = 1 if action == max_action else 0

        return self.current_policy

    ##### Policy Iteration #####
    def policyIterationV(self, config):
        # Initialize the value function and policy. We initialize the value function and policy here.
        self.initializeValueFunction(config)
        self.initializePolicy(config)
        self.newV = self.currentV.copy()
        converged = False
        
        logger.info(f"Starting policy iteration with {config.max_iterations} max iterations")
        count = 0

        while not converged:
            #logger.info(f"Iteration {i+1}")
            #### Policy evaluation ####
            for iter in range(config.max_iterations):
                oldV = self.currentV.copy()
                newV = self.PolicyEvaluationcalculateValueFunction()
                
                # Update currentV before checking convergence
                self.currentV = newV.copy()
                
                # Check convergence of value function
                currentVnp = np.array(list(oldV.values()))
                newVnp = np.array(list(self.currentV.values()))
                if np.allclose(currentVnp, newVnp, atol=config.epsilon):
                    logger.info(f"Value function converged after {iter+1} evaluation iterations")
                    break

            # Plot the value function after policy evaluation (For problem 2 and 1)
            if config.plotTable and count == 0 and (config.problem == 2 or config.problem == 1):
                temp_str = f"Values for each state after policy evaluation is complete"
                self.plotValueFunction(self.currentV, temp_str, config.problem)
            
            #### Policy improvement ####
            old_policy = {state: policy.copy() for state, policy in self.current_policy.items()}
            self.new_policy = self.policyImprovement()
            
            # Check if policy has changed
            policy_changed = False
            for state in self.states:
                if state not in self.terminal_states:
                    for action in self.actions:
                        if abs(old_policy[state][action] - self.new_policy[state][action]) > config.epsilon:
                            policy_changed = True
                            break
                    if policy_changed:
                        break
            
            self.current_policy = self.new_policy.copy()
            # Plot the value function and policy after the policy improvement (For problem 3)
            if config.plotTable and config.problem == 3:
                temp_str = f"Values and action for each state after {count+1} policy improvement"
                self.plotOptimalPolicy(self.currentV, self.current_policy, temp_str, config.problem, count+1)
            
            if not policy_changed:
                logger.info(f"Policy converged after {count+1} iterations")
                converged = True
                break
            count += 1

        if not converged:
            logger.info(f"Policy iteration did not converge after {config.max_iterations} iterations")
        else:
            logger.info(f"Policy iteration converged successfully")
        
        # Plot the value function and policy after the algorithm is complete (For problem 3)
        if config.plotTable and config.problem == 3:
            self.plotValueFunction(self.currentV, "Final Value Function after the algorithm is complete", config.problem)
            self.plotOptimalPolicy(self.currentV, self.current_policy, "Final Policy after the algorithm is complete", config.problem)

        return self.current_policy

    ##### Get Output Deterministic Policy #####
    def getOutputDeterministicPolicy(self):
        # get next states for each state
        policy = {state: {action: 0 for action in self.actions} for state in self.states}
        for state in self.states:
            if state in self.terminal_states:
                continue
                
            # Calculate V-value for each action and find the best one
            max_v_value = float('-inf')
            best_actions = []
            
            for action in self.actions:
                _, _, nextState, reward, _ = self.step(state, action)
                v_value = reward + self.gamma * self.currentV[nextState]
                
                if v_value > max_v_value:
                    max_v_value = v_value
                    best_actions = [action]
                elif v_value == max_v_value:
                    best_actions.append(action)
            
            # If multiple actions have the same V-value, prefer terminal states
            if len(best_actions) > 1:
                terminal_actions = [a for a in best_actions if self.getNextStates(state)[a] in self.terminal_states]
                if terminal_actions:
                    best_actions = terminal_actions
            
            max_action = best_actions[0]
            policy[state] = {action: 0 for action in self.actions}
            policy[state][max_action] = 1
        return policy

    ##### Value Iteration #####
    def valueIterationV(self, config):
        # Initialize the value function. We only initialize the value function here.
        self.initializeValueFunction(config)
        converged = False
        
        for iter in range(config.max_iterations):
            delta = 0
            for state in self.states:
                if state in self.terminal_states:
                    self.newV[state] = 0
                    continue
                # Calculate value for each action and take the maximum
                max_value = float('-inf')
                for action in self.actions:
                    _, _, nextState, reward, _ = self.step(state, action)
                    action_value = reward + self.gamma * self.currentV[nextState]
                    max_value = max(max_value, action_value)
                self.newV[state] = max_value
                delta = max(delta, math.fabs(self.currentV[state] - self.newV[state]))
            self.currentV = self.newV.copy()
            if delta < config.epsilon:
                logger.info(f"Value iteration converged after {iter+1} iterations")
                converged = True
                break
        
        optimal_policy = self.getOutputDeterministicPolicy()

        if not converged:
            logger.info(f"Value iteration did not converge after {config.max_iterations} iterations")
        else:
            logger.info(f"Value iteration converged successfully")

        # Plot the value function and policy after the algorithm is complete (For problem 4)
        if config.plotTable and config.problem == 4:
            self.plotValueFunction(self.currentV, "Final Value Function after the algorithm is complete", config.problem)
            self.plotOptimalPolicy(self.currentV, optimal_policy, "Final optimal policy after the algorithm is complete", config.problem, None)

        return optimal_policy

    # Plot code is generated by VSCode - GitHub Copilot
    def plotOptimalPolicy(self, valueFunction: dict, policy: dict, title: str = "Optimal Policy", problem: int = None, count: int = None):
        logger.info(f"Plotting optimal policy with values and actions")
        
        # Create a grid to store the display text for each cell
        display_grid = []
        for row in range(self.grid_size):
            display_row = []
            for col in range(self.grid_size):
                state = (row, col)
                
                # Get the value for this state
                value = valueFunction.get(state, 0.0)
                
                # Calculate state number (row * grid_size + col)
                state_number = row * self.grid_size + col
                
                # Get the optimal action for this state
                if state in self.terminal_states:
                    action_text = "TERM"
                elif self.spl_states is not None and state in self.spl_states:
                    # Find the action with probability 1 (optimal action) for special states
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
                        action_symbol = action_symbols.get(optimal_action, optimal_action)
                        action_text = action_symbol
                    else:
                        action_text = "?"
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
                
                # Combine value, action and state number in the cell
                if state in self.terminal_states:
                    cell_text = f"{value:.2f}\n{action_text}\n\nState {state_number} - TERM"
                elif self.spl_states is not None and state in self.spl_states:
                    cell_text = f"{value:.2f}\n{action_text}\n\nState {state_number} - SPL"
                else:
                    cell_text = f"{value:.2f}\n{action_text}\n\nState {state_number}"
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
        
        # Color code terminal states and special states
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = (row, col)
                if state in self.terminal_states:
                    table[(row, col)].set_facecolor('#90EE90')  # Light green for terminal states
                elif self.spl_states is not None and state in self.spl_states:
                    table[(row, col)].set_facecolor('#FFB6C1')  # Light pink for special states
                else:
                    table[(row, col)].set_facecolor('#F0F0F0')  # Light gray for regular states
        
        # Set smaller font for state numbers by modifying text properties
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = table[(row, col)]
                # Get the current text
                current_text = cell.get_text().get_text()
                # Split the text into lines
                lines = current_text.split('\n')
                if len(lines) > 0 and lines[-1].startswith('State '):
                    # Create new text with smaller font for state number using HTML-like formatting
                    main_text = '\n'.join(lines[:-1])  # All lines except the last
                    state_text = lines[-1]  # Last line (state number)
                    # Use a smaller font size for the state number
                    cell.get_text().set_text(f"{main_text}\n{state_text}")
                    # Set smaller font size for the entire cell to make state number smaller
                    cell.get_text().set_fontsize(10)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#90EE90', label='Terminal States'),
            plt.Rectangle((0,0),1,1, facecolor='#F0F0F0', label='Regular States')
        ]
        if self.spl_states is not None:
            legend_elements.insert(1, plt.Rectangle((0,0),1,1, facecolor='#FFB6C1', label='Special States'))
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()

        plt.savefig(f"images/prob{problem}{count}.png")
        plt.show()
        plt.close()

    # Plot code is generated by VSCode - GitHub Copilot
    def plotValueFunction(self, valueFunction: dict, title: str = "Value Function", problem: int = None):
        logger.info(f"Plotting value function")
        
        # Create a grid to store the display text for each cell
        display_grid = []
        for row in range(self.grid_size):
            display_row = []
            for col in range(self.grid_size):
                state = (row, col)
                
                # Get the value for this state
                value = valueFunction.get(state, 0.0)
                
                # Calculate state number (row * grid_size + col)
                state_number = row * self.grid_size + col
                
                # Format the value with appropriate precision and add state type indicator
                if abs(value) < 0.01:
                    value_text = f"{value:.3f}"
                else:
                    value_text = f"{value:.2f}"
                
                # Add state type indicator
                if state in self.terminal_states:
                    cell_text = f"{value_text}\n\nState {state_number} - TERM"
                elif self.spl_states is not None and state in self.spl_states:
                    cell_text = f"{value_text}\n\nState {state_number} - SPL"
                else:
                    cell_text = f"{value_text}\n\nState {state_number}"
                
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
        
        # Color code terminal states, special states, and use color gradient for values
        max_value = max(valueFunction.values()) if valueFunction.values() else 0
        min_value = min(valueFunction.values()) if valueFunction.values() else 0
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = (row, col)
                value = valueFunction.get(state, 0.0)
                
                if state in self.terminal_states:
                    table[(row, col)].set_facecolor('#90EE90')  # Light green for terminal states
                elif self.spl_states is not None and state in self.spl_states:
                    table[(row, col)].set_facecolor('#FFB6C1')  # Light pink for special states
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
        
        # Set smaller font for state numbers by modifying text properties
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = table[(row, col)]
                # Get the current text
                current_text = cell.get_text().get_text()
                # Split the text into lines
                lines = current_text.split('\n')
                if len(lines) > 0 and lines[-1].startswith('State '):
                    # Create new text with smaller font for state number using HTML-like formatting
                    main_text = '\n'.join(lines[:-1])  # All lines except the last
                    state_text = lines[-1]  # Last line (state number)
                    # Use a smaller font size for the state number
                    cell.get_text().set_text(f"{main_text}\n{state_text}")
                    # Set smaller font size for the entire cell to make state number smaller
                    cell.get_text().set_fontsize(10)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#90EE90', label='Terminal States'),
            plt.Rectangle((0,0),1,1, facecolor='#F0F0F0', label='Regular States')
        ]
        if self.spl_states is not None:
            legend_elements.insert(1, plt.Rectangle((0,0),1,1, facecolor='#FFB6C1', label='Special States'))
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        plt.savefig(f"images/prob{problem}.png")
        plt.show()
        plt.close()


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
    parser.add_argument("--uniformPolicyInit", action="store_true", help="Initialize the policy with uniform distribution")
    parser.add_argument("--problem", type=int, required=False, choices=[1, 2, 3 ,4], help="Problem number")
    parser.add_argument("--plotTable", action="store_true", help="Plot the value function and policy")
    parser.add_argument("--goalStates", type=tuple, default=[(0, 0), (3, 3)], help="Goal states list. Format list of tuples [(x, y), (x, y), ...]")
    parser.add_argument("--splStates", type=tuple, default=None, help="Spl states list. Format list of tuples [(x, y), (x, y), ...]")
    parser.add_argument("--splReward", type=int, default=None, help="Special state reward")
    config = parser.parse_args()

    if config.problem == 1:
        # Prob 1
        logger.info(f"Performing calculations for Prob 1: {config.task} with {config.valueFunctionInit} value function and uniform distribution for policy initialization")
        config.stepReward = -1
        config.goalReward = 0
        config.gamma = 0.9
        config.epsilon = 1e-6
        config.max_iterations = 200
        config.grid_size = 4
        config.valueFunctionInit = "V"
        config.randomValueFunctionInit = True
        config.uniformPolicyInit = True
        config.task = "policy_iteration"
        config.plotTable = True
        config.goalStates = [(0, 0), (3, 3)]
        config.splStates = None
        config.splReward = None
    elif config.problem == 2:
        # Prob 2
        logger.info(f"Performing calculations for Prob 2: {config.task} with {config.valueFunctionInit} value function and uniform distribution for policy initialization")
        config.stepReward = -4
        config.goalReward = 0
        config.gamma = 0.9
        config.epsilon = 1e-6
        config.max_iterations = 200
        config.grid_size = 4
        config.valueFunctionInit = "V"
        config.randomValueFunctionInit = True
        config.uniformPolicyInit = True
        config.task = "policy_iteration"
        config.plotTable = True
        config.goalStates = [(0, 0), (3, 3)]
        config.splStates = [(2,2)]
        config.splReward = -1
    elif config.problem == 3:
        # Prob 3
        logger.info(f"Performing calculations for Prob 3: {config.task} with {config.valueFunctionInit} value function and uniform distribution for policy initialization")
        config.stepReward = -4
        config.goalReward = 0
        config.gamma = 0.9
        config.epsilon = 1e-6
        config.max_iterations = 200
        config.grid_size = 4
        config.valueFunctionInit = "V"
        config.randomValueFunctionInit = True
        config.uniformPolicyInit = True
        config.task = "policy_iteration"
        config.plotTable = True
        config.goalStates = [(0, 0), (3, 3)]
        config.splStates = [(2,2)]
        config.splReward = -1
    elif config.problem == 4:
        # Prob 4
        logger.info(f"Performing calculations for Prob 4: {config.task} with {config.valueFunctionInit} value function and uniform distribution for policy initialization")
        config.stepReward = -4
        config.goalReward = 0
        config.gamma = 0.9
        config.epsilon = 1e-6
        config.max_iterations = 200
        config.grid_size = 4
        config.valueFunctionInit = "V"
        config.randomValueFunctionInit = True
        config.uniformPolicyInit = True
        config.task = "value_iteration"
        config.plotTable = True
        config.goalStates = [(0, 0), (3, 3)]
        config.splStates = [(2,2)]
        config.splReward = -1

    # initialize the grid world
    grid_world = GridWorld(config)

    if config.valueFunctionInit == "V":
        if config.task == "policy_iteration":
            optimal_policy = grid_world.policyIterationV(config)
            #logger.info(f"Optimal policy: {optimal_policy}")
        elif config.task == "value_iteration":
            grid_world.valueIterationV(config)

    if config.valueFunctionInit == "Q":
        logger.info(f"Q value function is not supported yet")

if __name__ == "__main__":
    main()