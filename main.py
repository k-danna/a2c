#!/usr/bin/env python3

import gym
from model import model
from worker import worker, replay_memory

def main():
    env = gym.make('CartPole-v0')
    #env = gym.make('Pong-v0')
    
    #since we are preprocessing state
    state_shape = worker(None).process_state(env.reset()).shape
    agent = worker(model(state_shape, env.action_space.n))
    agent.train(env)
    agent.test(env)

if __name__ == '__main__':
    main()
