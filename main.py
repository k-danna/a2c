#!/usr/bin/env python3

import gym
from model import model
from worker import worker, replay_memory

def main():
    env = gym.make('CartPole-v0')
    #env = gym.make('MountainCar-v0')
    #env = gym.make('Acrobot-v1')
    #env = gym.make('PongDeterministic-v4')
    env.seed(0)
    
    #since we are preprocessing state
    #n_actions = env.action_space.shape[0] #continuous (box env)
    n_actions = env.action_space.n #discrete env
    state_shape = worker(None).process_state(env.reset()).shape
    agent = worker(model(state_shape, n_actions))
    agent.train(env, episodes=10000, print_interval=1000)
    agent.test(env, episodes=100, print_interval=10, records=0)

if __name__ == '__main__':
    main()
