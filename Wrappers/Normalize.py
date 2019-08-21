# OpenAI Gym classes
import gym
from gym.core import Wrapper

# A Class to normalize state and rewards at will
class Normalize(Wrapper):
    def __init__(self, env):
        super(Normalize, self).__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation) 

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        # Punish and discourage terminal states or info['ale.lives'] < 3
        if done:
            reward += -1.0
            done = True

        # Add a living cost
        return self.observation(observation), self.reward(reward) - .01, done, info

    def reward(self, reward):
        return (reward) 

    def observation(self, observation):
        return observation / 256.0

