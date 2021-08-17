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


def generate_history_no_history():

        
    history = np.zeros((2,2))

    history[0, np.random.randint(2)] = 1
    
    history[1, np.random.randint(2)] = 1
        
    return history

def is_t4t(agent, n_samples):
    
    agent_actions = []
    t4t_actions = []
    
    for i in range(n_samples):
        history, n_turn = generate_history()
        action = agent.compute_action(history.flatten())
        if n_turn == 0:
            opponents_last_action = 0
        else:
            # opponents_last_action = np.where(history[1,:,n_turn-1])
            opponents_last_action = int(np.where(history[1,:,0])[0])
            
        agent_actions.append(action)
        t4t_actions.append(opponents_last_action)
        
    same_action = [int(agent_actions[i]==t4t_actions[i]) for i in range(n_samples)]
    t4t_fraction = np.sum(same_action)/n_samples,
    defect_fraction = np.sum([int(agent_action) for agent_action in agent_actions])/n_samples
    coop_fraction = 1 - defect_fraction
    
    return t4t_fraction, coop_fraction

def is_t4t_no_history(agent, n_samples):
    
    agent_actions = []
    t4t_actions = []
    
    for i in range(n_samples):
        history = generate_history_no_history()
        action = agent.compute_action(history.flatten())

        opponents_last_action = int(np.where(history[1,:])[0])
            
        agent_actions.append(action)
        t4t_actions.append(opponents_last_action)
        
    same_action = [int(agent_actions[i]==t4t_actions[i]) for i in range(n_samples)]
    t4t_fraction = np.sum(same_action)/n_samples,
    defect_fraction = np.sum([int(agent_action) for agent_action in agent_actions])/n_samples
    coop_fraction = 1 - defect_fraction
    
    return t4t_fraction, coop_fraction
    

class test_t4t_agent():
    
    def __init__(self, n_history=100, flip_frac=0):
        
        self.n_history = n_history
        self.flip_frac = flip_frac
    
    
    def compute_action(self, history):

        # if (not history.shape[0] == 2) or (not history.shape[1]==2):
        history = np.reshape(history, (2,2,-1))
            # print('reshaping')
        
        if sum(history[1, :, 0]) == 0:
            action = 0
        else:
            action = np.where(history[1, :, 0])[0]
            
        action = int(action)
        if np.random.rand() < self.flip_frac:
            action = int(not action)
        
        return action
        

class test_random_agent():
    
    def __init__(self, n_history=100, coop_frac=0.5):
        
        self.n_history = n_history
        self.coop_frac = coop_frac
    
    
    def compute_action(self, history):
        
        action = int(np.random.rand() > self.coop_frac)
        
        return action
        