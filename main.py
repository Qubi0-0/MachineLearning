from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt


class Actions(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class SimulationEnvironment:
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

    def simulate_episode(self):
        state = 1
        total_reward = 0
        steps_to_goal = 0

        for _ in range (self.max_actions):
            chosen_action = self.choose_random_action()
            current_state = self.state_transition(state, chosen_action)
            total_reward += self.reward(current_state)
            steps_to_goal += 1

            if current_state == self.goal_state:
                return total_reward, steps_to_goal

            state = current_state

        return total_reward, steps_to_goal

    def run(self):
        episode_rewards = []
        episode_steps_to_goal = []

        for _ in range(self.num_episodes):
            total_reward, steps_to_goal = self.simulate_episode()
            episode_rewards.append(total_reward)
            episode_steps_to_goal.append(steps_to_goal)

        return episode_rewards, episode_steps_to_goal

    def calc_and_plot(self, episode_rewards, episode_steps_to_goal):
        episode_rewards_std = np.std(episode_rewards)
        episode_steps_to_goal_std = np.std(episode_steps_to_goal)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_steps_to_goal_mean = np.mean(episode_steps_to_goal)
        average_reward_per_step = [r / s for r, s in zip(episode_rewards, episode_steps_to_goal)]

        print("Average Reward per Step:", average_reward_per_step)
        print("Average Steps to Goal:", episode_steps_to_goal_mean)
        print("Standard Deviation of Steps to Goal:", episode_steps_to_goal_std)
        print("Average Reward:", episode_rewards_mean)
        print("Standard Deviation of Reward:", episode_rewards_std)

        plt.figure(figsize=(10, 6))

        plt.subplot(1, 3, 1)
        plt.boxplot(episode_rewards)
        plt.title('Reward Distribution')

        plt.subplot(1, 3, 2)
        plt.boxplot(episode_steps_to_goal)
        plt.title('Steps to Goal Distribution')

        plt.subplot(1, 3, 3)
        plt.boxplot(average_reward_per_step)
        plt.title('Average Reward per Step Distribution')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    env = SimulationEnvironment()
    episode_rewards, episode_steps_to_goal = env.run()
    env.calc_and_plot(episode_rewards, episode_steps_to_goal)
