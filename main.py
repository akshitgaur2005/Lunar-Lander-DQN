import gymnasium as gym
import random
from collections import deque, namedtuple
import numpy as np
import torch
from torch import nn, optim

def main():
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    state, info = env.reset()

    num_episodes = 100
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    agent = DQN(n_obs, n_actions)

    event = namedtuple("Event", "state, action, reward, next_state")

    for i in range(num_episodes):
        print(i)
        term, trunc = False, False
        while not (term or trunc):
            action = agent.get_action(state)
            #print(action)
            next_state, reward, term, trunc, info = env.step(action)
            agent.memory.append(event(state, action, reward, next_state))
            state = next_state

    states, actions, rewards, next_states = list(zip(*agent.memory))
    print(np.array(next_states).shape)

class DQN():

    def __init__(self, n_obs, n_actions):
        self.memory = deque()
        self.model = nn.Sequential(
                nn.LazyLinear(64), nn.ReLU(),
                nn.LazyLinear(16), nn.ReLU(),
                nn.LazyLinear(n_actions), nn.ReLU(),
                )

    def get_action(self, state):
        return random.randrange(4)

if __name__ == "__main__":
    main()
