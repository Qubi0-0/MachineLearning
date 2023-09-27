from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt


class Actions(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Environment:
    def __init__(self, num_episodes=30, max_actions=1000, num_states=100, goal_state=100):
        self.num_episodes = num_episodes
        self.max_actions = max_actions
        self.num_states = num_states
        self.goal_state = goal_state

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
        if state == self.goal_state:
            return 100
        else:
            return 0

class Agent: 
    def __init__(self):
        pass




if __name__ == "__main__":
    env = Environment()
