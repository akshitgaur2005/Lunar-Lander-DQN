import gymnasium as gym
import random
from collections import deque, namedtuple
import numpy as np

def main():
    env = gym.make("LunarLander-v2", render_mode="human")

    state, info = env.reset()

    num_episodes = 10
    agent = DQN()

    event = namedtuple("Event", "state, action, reward, next_state")

    for i in range(num_episodes):
        action = agent.get_action(state)
        print(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.memory.append(event(state, action, reward, next_state))
        next_state = state

    states, actions, rewards, next_states = list(zip(*agent.memory))
    print(np.array(states))

class DQN():

    def __init__(self):
        self.memory = deque()

    def get_action(self, state):
        return random.randrange(4)

if __name__ == "__main__":
    main()
