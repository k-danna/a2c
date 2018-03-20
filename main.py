#!/usr/bin/env python3

import gym
from model import model
from worker import worker, replay_memory

def main():
    env = gym.make('CartPole-v0')
    #env = gym.make('Pong-v0')
    env.seed(0) #performs well (200 mean, 0 std)
    #env.seed(42) #performs poor (9.29 mean, 0.795 std)
    
    #since we are preprocessing state
    state_shape = worker(None).process_state(env.reset()).shape
    agent = worker(model(state_shape, env.action_space.n))
    agent.train(env, episodes=10000)
    agent.test(env)

if __name__ == '__main__':
    main()
