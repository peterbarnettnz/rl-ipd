import numpy as np
import torch
import matplotlib.pyplot as plt

import gym

from gym import spaces

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG as DEFAULT_CONFIG_PPO

from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG 
from ray.rllib.agents.dqn import  DEFAULT_CONFIG as DEFAULT_CONFIG_DQN


from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray import tune



def generate_history(n_history=100, n_turn=None):
    if n_turn is None:
        n_turn = np.random.randint(n_history)
        
    history = np.zeros((2,2,n_history))
    
    for i in range(n_turn):
        history[0, np.random.randint(2), i] = 1
        
        history[1, np.random.randint(2), i] = 1
        
    return history, n_turn

def is_t4t(agent, n_samples):
    
    agent_actions = []
    t4t_actions = []
    
    for i in range(n_samples):
        history, n_turn = generate_history()
        action = agent.compute_action(history)
        if n_turn == 0:
            opponents_last_action = 0
        else:
            # opponents_last_action = np.where(history[1,:,n_turn-1])
            opponents_last_action = np.where(history[1,:,0])
            
        agent_actions.append(action)
        t4t_actions.append(opponents_last_action)
        
    same_action = [int(agent_actions[i]==t4t_actions[i]) for i in range(n_samples)]
    t4t_fraction = np.sum(same_action)/n_samples
    
    return t4t_fraction
    

class test_t4t_agent():
    
    def __init__(self, n_history=100, flip_frac=0):
        
        self.n_history = n_history
        self.flip_frac = flip_frac
    
    
    def compute_action(self, history):
        
        if sum(history[1, :, 0]) == 0:
            action = 0
        else:
            action = np.where(history[1, :, 0])
            
        if np.random.rand() < self.flip_frac:
            action = int(not action)
        
        return action
        