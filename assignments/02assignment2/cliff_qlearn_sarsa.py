import os
import sys
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GridWorld class
class GridWorld:
    def __init__(self, config):
        self.config = config
        self.grid_height = config.grid_size[0]
        self.grid_width = config.grid_size[1]
        self.grid = np.zeros((self.grid_height, self.grid_width))
        self.states = [(i, j) for i in range(self.grid_height) for j in range(self.grid_width)]
        self.goal_states = config.goal_states
        self.start_state = tuple(config.start_state[0]) if isinstance(config.start_state[0], list) else config.start_state[0]
        self.cliff_states = config.cliff_states
        self.cliff_reward = config.cliff_reward
        self.step_reward = config.step_reward
        self.goal_reward = config.goal_reward
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.max_episodes = config.max_episodes
        self.max_steps_per_episode = config.max_steps_per_episode
        self.actions = ["up", "down", "left", "right"]
        self.Q_qlearning = self.initialize_Q_qlearning()
        self.Q_sarsa = self.initialize_Q_sarsa()
        self.alpha_qlearning = config.alpha_qlearning
        self.alpha_sarsa = config.alpha_sarsa

    # Initialize Q-learning Q-table (numpy is faster but dictionary is easier to understand)
    def initialize_Q_qlearning(self):
        return {state: {action: 0 for action in self.actions} for state in self.states}

    # Initialize SARSA Q-table
    def initialize_Q_sarsa(self):
        return {state: {action: 0 for action in self.actions} for state in self.states}

    # Step function
    def step(self, state, action):
        # If already at goal, stay there
        if state in self.goal_states:
            return state, self.goal_reward, True
            
        # Calculate next state based on action
        if action == "up":
            next_state = (state[0] - 1, state[1])
        elif action == "down":
            next_state = (state[0] + 1, state[1])
        elif action == "left":
            next_state = (state[0], state[1] - 1)
        elif action == "right":
            next_state = (state[0], state[1] + 1)
        else:
            next_state = state

        # Check boundaries - if out of bounds, stay in current state
        if next_state not in self.states:
            next_state = state

        # Determine reward and done status
        if next_state in self.cliff_states:
            # Hit a cliff - reset to start and give cliff penalty
            next_state = self.start_state
            reward = self.cliff_reward
            done = False
        elif next_state in self.goal_states:
            # Reached goal
            reward = self.goal_reward
            done = True
        else:
            # Normal step
            reward = self.step_reward
            done = False
            
        return next_state, reward, done

    # Although the policy functions are the same for Q-learning and SARSA, 
    # I am keeping them separate for my own clarity.
    # Get action epsilon-greedy
    def get_action_epsilon_greedy(self, state):
        # If random number is less than epsilon, return random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            # Otherwise, return action with highest Q-value
            return self.get_action_max_Q_sarsa(state)

    # Get action with highest Q-value - Greedy policy
    def get_action_max_Q_sarsa(self, state):
        # Return action with highest Q-value
        return max(self.Q_sarsa[state], key=self.Q_sarsa[state].get)

    # SARSA algorithm
    def sarsa(self):
        Q_sarsa = []
        episode_rewards = []
        
        # Run for maximum number of episodes
        for episode in range(self.max_episodes):
            # Initialize current state to start state
            current_state = self.start_state
            # Get action epsilon-greedy
            current_action = self.get_action_epsilon_greedy(current_state)
            # Initialize episode reward
            episode_reward = 0
            
            # Run for maximum number of steps per episode
            for step in range(self.max_steps_per_episode):
                state = current_state
                # Initialize action to current action
                action = current_action
                # Take step
                next_state, reward, done = self.step(state, action)
                # Get action epsilon-greedy - on policy. So still epsilon-greedy policy for next action.
                next_action = self.get_action_epsilon_greedy(next_state)
                
                # SARSA update: use actual next action
                if done:
                    # Terminal state has Q-value 0
                    next_q_value = 0
                else:
                    next_q_value = self.Q_sarsa[next_state][next_action]
                self.Q_sarsa[state][action] = self.Q_sarsa[state][action] + self.alpha_sarsa * (reward + self.gamma * next_q_value - self.Q_sarsa[state][action])
                
                # total reward for episode for plotting
                episode_reward += reward
                # Update current state to next state
                current_state = next_state
                # Update current action to next action
                current_action = next_action
                
                # If done, break
                if done:
                    #logger.info(f"Episode {episode} completed in {step} steps with reward {episode_reward}")
                    break

            # Append episode reward to list for plotting
            episode_rewards.append(episode_reward)
            
            # Store Q-table every 100 episodes for plotting
            # This is not necessary for SARSA, but is useful for plotting.
            if episode % 100 == 0:
                Q_sarsa.append(self.Q_sarsa.copy())
                
        return self.Q_sarsa, Q_sarsa, episode_rewards

    # Get action epsilon-greedy
    def get_action_epsilon_greedy_qlearning(self, state):
        # If random number is less than epsilon, return random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            # Otherwise, return action with highest Q-value
            return self.get_action_max_Q(state)

    # Get action with highest Q-value - Greedy policy
    def get_action_max_Q(self, state):
        # Return action with highest Q-value
        return max(self.Q_qlearning[state], key=self.Q_qlearning[state].get)

    # Q-learning algorithm
    def Qlearning(self):
        # Q list for plotting and episode rewards for convergence
        Q_qlearning = []
        episode_rewards = []
        
        # Run for maximum number of episodes
        for episode in range(self.max_episodes):
            # Initialize current state to start state
            current_state = self.start_state
            episode_reward = 0
            
            # Run for maximum number of steps per episode
            for step in range(self.max_steps_per_episode):
                # Initialize state to current state
                state = current_state
                # Get action epsilon-greedy
                action = self.get_action_epsilon_greedy_qlearning(state)
                # Take step
                next_state, reward, done = self.step(state, action)
                
                # Q-learning update: use max Q-value for next state
                if done:
                    max_q_next = 0
                else:
                    max_q_next = max(self.Q_qlearning[next_state].values())
                self.Q_qlearning[state][action] = self.Q_qlearning[state][action] + self.alpha_qlearning * (reward + self.gamma * max_q_next - self.Q_qlearning[state][action])
                
                # total reward for episode for plotting
                episode_reward += reward
                current_state = next_state
                
                # If done, break
                if done:
                    #logger.info(f"Episode {episode} completed in {step} steps with reward {episode_reward}")
                    break
            
            episode_rewards.append(episode_reward)
            
            # Store Q-table every 100 episodes for plotting
            if episode % 100 == 0:
                Q_qlearning.append(self.Q_qlearning.copy())
                
        return self.Q_qlearning, Q_qlearning, episode_rewards

    def simulate_path(self, Q_table, max_steps=100):
        path = [self.start_state]
        current_state = self.start_state
        step_count = 0
        
        while current_state not in self.goal_states and step_count < max_steps:
            # Get best action according to Q-table
            best_action = max(Q_table[current_state], key=Q_table[current_state].get)
            
            # Take the action
            next_state, reward, done = self.step(current_state, best_action)
            path.append(next_state)
            current_state = next_state
            step_count += 1
            
            if done:
                break
        
        return path

    def plot_gridworld_with_path(self, Q_sarsa=None, Q_qlearning=None, show_sarsa_path=True, show_qlearning_path=True, gam=None, eps=None):
        # Create a color map for different types of states
        grid_colors = np.zeros((self.grid_height, self.grid_width, 3))
        
        # Default color for regular states (light gray)
        grid_colors.fill(0.9)
        
        # Color for start state (green)
        start_row, start_col = self.start_state
        grid_colors[start_row, start_col] = [0.2, 0.8, 0.2]
        
        # Color for goal states (blue)
        for goal_state in self.goal_states:
            goal_row, goal_col = goal_state
            grid_colors[goal_row, goal_col] = [0.2, 0.2, 0.8]
        
        # Color for cliff states (red)
        for cliff_state in self.cliff_states:
            cliff_row, cliff_col = cliff_state
            grid_colors[cliff_row, cliff_col] = [0.8, 0.2, 0.2]
        
        # Calculate figure size based on grid dimensions
        fig_width = max(8, self.grid_width * 1.5)
        fig_height = max(8, self.grid_height * 1.5)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Display the grid
        ax.imshow(grid_colors, interpolation='nearest', aspect='equal')
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        
        # Set ticks and labels
        ax.set_xticks(range(self.grid_width))
        ax.set_yticks(range(self.grid_height))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title('GridWorld Environment with Learned Paths')
        
        # Add text annotations for each cell
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_text = ""
                if (i, j) == self.start_state:
                    cell_text = "S"
                elif (i, j) in self.goal_states:
                    cell_text = "G"
                elif (i, j) in self.cliff_states:
                    cell_text = "C"
                else:
                    cell_text = ""
                
                ax.text(j, i, cell_text, ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='white' if cell_text else 'black')
        
        # Plot SARSA path if requested
        if show_sarsa_path and Q_sarsa is not None:
            sarsa_path = self.simulate_path(Q_sarsa)
            if len(sarsa_path) > 1:
                sarsa_x = [pos[1] - 0.1 for pos in sarsa_path]
                sarsa_y = [pos[0] for pos in sarsa_path]
                ax.plot(sarsa_x, sarsa_y, 'b-', linewidth=4, alpha=0.8, label='SARSA Path')
                # Add arrows to show direction
                for i in range(len(sarsa_path)-1):
                    dx = sarsa_x[i+1] - sarsa_x[i]
                    dy = sarsa_y[i+1] - sarsa_y[i]
                    if dx != 0 or dy != 0:
                        ax.arrow(sarsa_x[i], sarsa_y[i], dx*0.3, dy*0.3, 
                               head_width=0.15, head_length=0.15, fc='blue', ec='blue', alpha=0.8)
        
        # Plot Q-learning path if requested
        if show_qlearning_path and Q_qlearning is not None:
            qlearning_path = self.simulate_path(Q_qlearning)
            if len(qlearning_path) > 1:
                qlearning_x = [pos[1] + 0.1 for pos in qlearning_path]
                qlearning_y = [pos[0] for pos in qlearning_path]
                ax.plot(qlearning_x, qlearning_y, 'r--', linewidth=4, alpha=0.8, label='Q-learning Path')
                # Add arrows to show direction
                for i in range(len(qlearning_path)-1):
                    dx = qlearning_x[i+1] - qlearning_x[i]
                    dy = qlearning_y[i+1] - qlearning_y[i]
                    if dx != 0 or dy != 0:
                        ax.arrow(qlearning_x[i], qlearning_y[i], dx*0.3, dy*0.3, 
                               head_width=0.15, head_length=0.15, fc='red', ec='red', alpha=0.8)
        
        # Create legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.2, 0.8, 0.2], label='Start (S)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.2, 0.2, 0.8], label='Goal (G)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.8, 0.2, 0.2], label='Cliff (C)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.9, 0.9, 0.9], label='Regular')
        ]
        
        # Add path legends if paths are shown
        if show_sarsa_path and Q_sarsa is not None:
            legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=4, label='SARSA Path', linestyle='-'))
        if show_qlearning_path and Q_qlearning is not None:
            legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=4, label='Q-learning Path', linestyle='--'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        # Invert y-axis to match typical grid coordinates (0,0 at top-left)
        ax.invert_yaxis()
        
        plt.tight_layout()
        #plt.show()
        gam_str = str(gam).replace('.', '_')
        eps_str = str(eps).replace('.', '_')
        img_name = f'images/gridworld_with_paths_{gam_str}_{eps_str}.png'
        if os.path.exists(img_name):
            img_name = f'images/2_gridworld_with_paths_{gam_str}_{eps_str}.png'
        logger.info(f"Saving gridworld with paths to {img_name}")
        plt.savefig(img_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print path information
        if show_sarsa_path and Q_sarsa is not None:
            sarsa_path = self.simulate_path(Q_sarsa)
            print(f"SARSA Path: {sarsa_path}")
            print(f"SARSA Path Length: {len(sarsa_path)} steps")
        
        if show_qlearning_path and Q_qlearning is not None:
            qlearning_path = self.simulate_path(Q_qlearning)
            print(f"Q-learning Path: {qlearning_path}")
            print(f"Q-learning Path Length: {len(qlearning_path)} steps")

    def plot_episode_rewards_gamma(self, sarsa_rewards, qlearning_rewards, gamma):
        # Create smoothed rewards using moving average
        window_size = 50
        sarsa_smooth = []
        qlearning_smooth = []
        
        for i in range(len(sarsa_rewards)):
            start_idx = max(0, i - window_size + 1)
            sarsa_smooth.append(np.mean(sarsa_rewards[start_idx:i+1]))
        
        for i in range(len(qlearning_rewards)):
            start_idx = max(0, i - window_size + 1)
            qlearning_smooth.append(np.mean(qlearning_rewards[start_idx:i+1]))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use normal episode indexing (0-based)
        episodes = np.arange(len(sarsa_rewards))
        
        # Plot raw rewards (thin, transparent)
        ax.plot(episodes, sarsa_rewards, 'b-', alpha=0.1, linewidth=0.5, label='SARSA (raw)')
        ax.plot(episodes, qlearning_rewards, 'r-', alpha=0.1, linewidth=0.5, label='Q-learning (raw)')
        
        # Plot smoothed rewards (thick, opaque)
        ax.plot(episodes, sarsa_smooth, 'b-', linewidth=2, label=f'SARSA (smoothed, window={window_size})')
        ax.plot(episodes, qlearning_smooth, 'r-', linewidth=2, label=f'Q-learning (smoothed, window={window_size})')
        
        # # Set log scale only for y-axis
        # ax.set_yscale('log')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title(f'Episode Rewards Over Time when gamma = {gamma}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at optimal reward
        optimal_reward = self.goal_reward + (6 * self.step_reward)
        ax.axhline(y=optimal_reward, color='green', linestyle='--', alpha=0.7, label=f'Optimal Reward ({optimal_reward})')
        ax.legend()
        
        plt.tight_layout()
        #plt.show()

        gam_str = str(gamma).replace('.', '_')
        img_name = f'images/episode_rewards_gamma_{gam_str}.png'
        if os.path.exists(img_name):
            img_name = f'images/2_episode_rewards_gamma_{gam_str}.png'
        logger.info(f"Saving episode rewards to {img_name}")
        plt.savefig(img_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print(f"\nEpisode Reward Statistics:")
        print(f"SARSA - Final 100 episodes avg: {np.mean(sarsa_rewards[-100:]):.2f}")
        print(f"Q-learning - Final 100 episodes avg: {np.mean(qlearning_rewards[-100:]):.2f}")
        print(f"Optimal reward: {optimal_reward}")

    def plot_episode_rewards_epsilon(self, sarsa_rewards, qlearning_rewards, epsilon):
        # Create smoothed rewards using moving average
        window_size = 50
        sarsa_smooth = []
        qlearning_smooth = []
        
        for i in range(len(sarsa_rewards)):
            start_idx = max(0, i - window_size + 1)
            sarsa_smooth.append(np.mean(sarsa_rewards[start_idx:i+1]))
        
        for i in range(len(qlearning_rewards)):
            start_idx = max(0, i - window_size + 1)
            qlearning_smooth.append(np.mean(qlearning_rewards[start_idx:i+1]))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use normal episode indexing (0-based)
        episodes = np.arange(len(sarsa_rewards))
        
        # Plot raw rewards (thin, transparent)
        ax.plot(episodes, sarsa_rewards, 'b-', alpha=0.1, linewidth=0.5, label='SARSA (raw)')
        ax.plot(episodes, qlearning_rewards, 'r-', alpha=0.1, linewidth=0.5, label='Q-learning (raw)')
        
        # Plot smoothed rewards (thick, opaque)
        ax.plot(episodes, sarsa_smooth, 'b-', linewidth=2, label=f'SARSA (smoothed, window={window_size})')
        ax.plot(episodes, qlearning_smooth, 'r-', linewidth=2, label=f'Q-learning (smoothed, window={window_size})')
        
        # # Set log scale only for y-axis
        # ax.set_yscale('log')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title(f'Episode Rewards Over Time when epsilon = {epsilon}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at optimal reward
        optimal_reward = self.goal_reward + (6 * self.step_reward)
        ax.axhline(y=optimal_reward, color='green', linestyle='--', alpha=0.7, label=f'Optimal Reward ({optimal_reward})')
        ax.legend()
        
        plt.tight_layout()
        #plt.show()
        eps_str = str(epsilon).replace('.', '_')
        img_name = f'images/episode_rewards_epsilon_{eps_str}.png'
        if os.path.exists(img_name):
            img_name = f'images/2_episode_rewards_epsilon_{eps_str}.png'
        logger.info(f"Saving episode rewards to {img_name}")
        plt.savefig(img_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print(f"\nEpisode Reward Statistics:")
        print(f"SARSA - Final 100 episodes avg: {np.mean(sarsa_rewards[-100:]):.2f}")
        print(f"Q-learning - Final 100 episodes avg: {np.mean(qlearning_rewards[-100:]):.2f}")
        print(f"Optimal reward: {optimal_reward}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=tuple, default=(4, 12), help="Size of the grid")
    parser.add_argument("--goal_states", type=list, default=[(0, 11)], help="Goal states")
    parser.add_argument("--start_state", type=list, default=[(0, 0)], help="Start state")
    parser.add_argument("--cliff_states", type=list, default=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10)], help="Cliff states")
    parser.add_argument("--cliff_reward", type=int, default=-100, help="Reward for falling into a cliff")
    parser.add_argument("--step_reward", type=int, default=-1, help="Reward for each step")
    parser.add_argument("--goal_reward", type=int, default=100, help="Reward for reaching the goal")
    parser.add_argument("--gamma", type=float, default=0.1, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy policy")
    parser.add_argument("--max_episodes", type=int, default=10000, help="Maximum number of episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--alpha_qlearning", type=float, default=0.1, help="Learning rate for Q-learning")
    parser.add_argument("--alpha_sarsa", type=float, default=0.1, help="Learning rate for SARSA")
    parser.add_argument("--run_all", action="store_true", help="Run all the experiments for gamma and epsilon")
    args = parser.parse_args()
    config = args

    logger.info(f"Config: {config}")

    if config.run_all:
        gamma_list = [0.01, 0.1, 0.5, 0.99, 1]
        epsilon_list = [0.01, 0.1, 0.5, 0.99]
    else:
        gamma_list = [config.gamma]
        epsilon_list = [config.epsilon]
    
    # Predefined configurations for assignment
    logger.info(f"Running Q learning and SARSA with Gamma list: {gamma_list}")
    for gamma in gamma_list:
        config.gamma = gamma
        gridworld = GridWorld(config)
        Q_sarsa, Q_sarsa_list, sarsa_rewards = gridworld.sarsa()
        Q_qlearning, Q_qlearning_list, qlearning_rewards = gridworld.Qlearning()
    
        # # Plot the gridworld environment with learned paths
        gridworld.plot_gridworld_with_path(Q_sarsa, Q_qlearning, gam=gamma, eps=config.epsilon)
        
        # Plot episode rewards
        gridworld.plot_episode_rewards_gamma(sarsa_rewards, qlearning_rewards, gamma)

    config.gamma = 0.99
    logger.info(f"Running Q learning and SARSA with Epsilon list: {epsilon_list}")
    for epsilon in epsilon_list:
        config.epsilon = epsilon
        gridworld = GridWorld(config)
        Q_sarsa, Q_sarsa_list, sarsa_rewards = gridworld.sarsa()
        Q_qlearning, Q_qlearning_list, qlearning_rewards = gridworld.Qlearning()

        # Plot the gridworld environment with learned paths
        gridworld.plot_gridworld_with_path(Q_sarsa, Q_qlearning, gam=config.gamma, eps=epsilon)

        # Plot episode rewards
        gridworld.plot_episode_rewards_epsilon(sarsa_rewards, qlearning_rewards, epsilon)

if __name__ == "__main__":
    main()