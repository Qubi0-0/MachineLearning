import random 
import numpy as np
from tqdm import tqdm
from enum import Enum
from matplotlib import pyplot as plt





class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class QLearning:
    def __init__(self, num_states=100, num_actions=4, alpha=0.7, gamma=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((num_states, num_actions))

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
        if state == 99:
            return 100
        else:
            return 0

    def update_q_table(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action.value] = (1 - self.alpha) * self.q_table[state, action.value] + \
            self.alpha * (reward + self.gamma * self.q_table[next_state, best_next_action])

    def find_best_action(self, state):
        return np.argmax(self.q_table[state, :])

    # def choose_action(self, state, epsilon=0.1):
    #     if random.random() < epsilon:
    #         return random.randint(0, self.num_actions - 1)  # Explore
    #     else:
    #         return np.argmax(self.q_table[state, :])  # Exploit
        
    def run(self, starting_state = 1, num_steps = 1000):
        state = starting_state
        for _ in tqdm(range(num_steps)):
            action = self.choose_random_action()
            reward = self.reward(state)
            action = self.choose_random_action()
            next_state = self.state_transition(state, action)
            if next_state == 100:
                next_state = 99
            self.update_q_table(state, action, next_state, reward)
            state = next_state


if __name__ == "__main__":
    q_learning = QLearning()
    q_learning.run( starting_state= 1, num_steps= 20000)

    best_action_matrix = np.argmax(q_learning.q_table, axis=1)

    # Create a heatmap of the maximum Q-values
    plt.figure(figsize=(10, 10))
    plt.imshow(np.max(q_learning.q_table, axis=1).reshape(10, 10), cmap='coolwarm', origin='lower', aspect='auto')
    plt.colorbar(label='Max Q-Value')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('Heatmap of Maximum Q-Values per State')
    plt.xticks(np.arange(0, 10))
    plt.yticks(np.arange(0, 10))
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.show()

    # Display the best action for each state
    print("Best Action for Each State:")
    for row in range(10):
        for col in range(10):
            state = row * 10 + col
            action = best_action_matrix[state]
            print(f"State {state}: Best Action - {Actions(action).name}")