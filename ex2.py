import random 
import numpy as np
from tqdm import tqdm
from enum import IntEnum
from matplotlib import pyplot as plt
import time

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class QLearning:
    def __init__(self, num_states= 100, num_actions= 4, alpha= 0.7, gamma= 0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((num_states, num_actions))
        self.stop_points = [100, 200, 500, 600, 700, 800, 900, 1000, 2500,
                             5000, 7500, 10000, 12500, 15000, 17500, 20000]

    def state_transition(self, state, action):
        if action == Actions.RIGHT and state % 10 != 0:
            return state + 1
        elif action == Actions.LEFT and (state - 1) % 10 != 0:
            return state - 1
        elif action == Actions.UP and state > 10:
            return state - 10
        elif action == Actions.DOWN and state <= 90:
            return state + 10
        else:
            return state
    
    def choose_random_action(self):
        return random.choice(list(Actions))

    def reward(self, state):
        if state == 100:
            return 100
        else:
            return 0

    def update_q_table(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action.value] = (1 - self.alpha) * self.q_table[state, action.value] + \
            self.alpha * (reward + self.gamma * self.q_table[next_state, best_next_action])

    def find_best_action(self, state):
        return np.argmax(self.q_table[state, :])

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)  # Explore
        else:
            max_q_value = np.max(self.q_table[state -1, :])
            best_actions = [action for action, q_value in enumerate(self.q_table[state -1, :]) if q_value == max_q_value]
            return random.choice(best_actions)  # Exploit among best actions

        
    def run(self, starting_state=1, num_steps=1000, update=True, random_action=True):
        reward = 0
        state = starting_state

        run_times = []  # List to store run-times
        rewards_per_experiment = []
        start_time = time.time()  # Record start time
        for num_step in (range(1, num_steps + 1)):
            if random_action:
                action = self.choose_random_action()
            else:
                action = self.choose_action(state)
            reward = self.reward(state)
            next_state = self.state_transition(state, action)
            if update:
                self.update_q_table(state - 1, action, next_state - 1, reward)
            state = next_state
            if num_step in self.stop_points:
                avg_reward = q_learning.test()
                rewards_per_experiment.append(avg_reward)
                end_time = time.time()  # Record end time
                run_time = end_time - start_time  # Calculate run-time
                run_times.append(run_time)
                start_time = time.time()  # Reset start time

        # Fill in missing reward values with zeros
        while len(rewards_per_experiment) < len(self.stop_points):
            rewards_per_experiment.append(0)

        return rewards_per_experiment, run_times


    def test(self):
        """A test consists of running the system for 1000 steps using the current Q table (without
            changing it) and always choosing the best action at each step. Measure the average
            reward per step in these 1000 steps."""
        state = 1
        rewards = []
        num_steps = 1000
        for _ in range(num_steps):
            action = self.choose_action(state)
            reward = self.reward(state)
            rewards.append(reward)
            next_state = self.state_transition(state, action)
            state = next_state

        avg_reward = np.mean(rewards)
        return avg_reward


def create_heatmap(heat_map, ax):
    ax.imshow(np.max(heat_map, axis=1).reshape(10, 10), cmap='hot', origin='upper', aspect='auto')
    ax.set_xticks(np.arange(0, 10))
    ax.set_yticks(np.arange(0, 10))
    ax.grid(visible=True, linestyle='--', alpha=0.5)

def plot_heatmaps(list_of_heatmaps, num_rows=5, num_cols=6):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 12))

    for i, heatmap in enumerate(list_of_heatmaps):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]

        create_heatmap(heatmap, ax)
        ax.set_title(f'Heatmap {i+1}')

    for i in range(len(list_of_heatmaps), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    q_learning = QLearning()
    num_experiments = 30  # Number of experiments
    avg_rewards = []
    run_times = []  # List to store run-times
    heat_maps = []
    for _ in tqdm(range(num_experiments)):
        q_learning.q_table = np.zeros((100, 4))
        rewards_per_experiment, runtime_per_experiment = q_learning.run(starting_state=1, num_steps=20000)
        avg_rewards.append(rewards_per_experiment)
        run_times.append(runtime_per_experiment)
        heat_maps.append(q_learning.q_table)

    # Plot heatmaps
    plot_heatmaps(heat_maps)

    avg_rewards = np.array(avg_rewards)
    print(avg_rewards)
    # Calculate the mean and standard deviation along the experiments axis
    mean_rewards = np.mean(avg_rewards, axis=0)
    std_dev_rewards = np.std(avg_rewards, axis=0)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot steps (x-axis) vs. average reward (y-axis) in the first subplot
    ax1.errorbar(q_learning.stop_points, mean_rewards, yerr=std_dev_rewards, marker='o', linestyle='-')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    # ax1.set_ylim(-0.0001, 0.0001)
    ax1.set_title('Steps vs. Average Reward')
    ax1.grid(visible=True, linestyle='--', alpha=0.5)

    # Create a box plot for run-times in the second subplot
    ax2.boxplot(run_times, vert=False)
    ax2.set_xlabel('Run-time (seconds)')
    ax2.set_title('Box Plot of Run-Times')
    ax2.grid(visible=True, linestyle='--', alpha=0.5)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

    # Print average and standard deviation of run-times
    avg_runtime = np.mean(run_times)
    std_dev_runtime = np.std(run_times)
    print(f"Average Run-time: {avg_runtime:.6f} seconds")
    print(f"Standard Deviation of Run-times: {std_dev_runtime:.6f} seconds")
