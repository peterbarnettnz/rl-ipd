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



def generate_history(n_history=100, n_turn=None, state_len=None):

    if state_len is None:
        state_len = n_history
    if n_turn is None:
        n_turn = np.random.randint(n_history)
        
    history = np.zeros((2,2,n_history))
    
    for i in range(n_turn):
        history[0, np.random.randint(2), i] = 1
        
        history[1, np.random.randint(2), i] = 1
        
    return history[:,:,:state_len], n_turn


def generate_history_no_history():

        
    history = np.zeros((2,2))

    history[0, np.random.randint(2)] = 1
    
    history[1, np.random.randint(2)] = 1
        
    return history

def is_t4t(agent, n_samples, policy_id='default_policy', n_history=100, state_len=None):
    
    if state_len is None:
        state_len=n_history

    agent_actions = []
    t4t_actions = []
    
    for i in range(n_samples):
        history, n_turn = generate_history(n_history=n_history, state_len=state_len)
        action = agent.compute_action(history.flatten(), policy_id=policy_id)
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
    
    return t4t_fraction[0], coop_fraction

def play_agents(agents, env, n_games=1):

    n_samples = 0

    t4t_sum0 = 0
    t4t_sum1 = 0
    coop_sum0 = 0
    coop_sum1 = 0

    for i_game in range(n_games):

        states = env.reset()
        done = False
        actions = {0:0, 1:0}

        while not done:

            action0 = agents.compute_action(states[0], policy_id='agent-0')            
            action1 = agents.compute_action(states[1], policy_id='agent-1')

            if action0 == actions[1]:
                t4t_sum0 += 1

            if action1 == actions[0]:
                t4t_sum1 += 1

            if action0 == 0:
                coop_sum0 += 1

            if action1 == 0:
                coop_sum1 += 1

            actions = {0: action0, 1:action1}
            
            states, rewards, dones, infos = env.step(actions)

            n_samples += 1


            if dones['__all__']:
                done = True

    t4t_frac0 = t4t_sum0 / n_samples
    t4t_frac1 = t4t_sum1 / n_samples
    coop_frac0 = coop_sum0 / n_samples
    coop_frac1 = coop_sum1 / n_samples

    return t4t_frac0, t4t_frac1, coop_frac0, coop_frac1
            
            



def is_t4t_no_history_old(agent, n_samples, policy_id='default_policy'):
    
    agent_actions = []
    t4t_actions = []
    
    for i in range(n_samples):
        history = generate_history_no_history()
        action = agent.compute_single_action(observation=history.flatten(), policy_id=policy_id)

        opponents_last_action = int(np.where(history[1,:])[0])
            
        agent_actions.append(action)
        t4t_actions.append(opponents_last_action)
        
    same_action = [int(agent_actions[i]==t4t_actions[i]) for i in range(n_samples)]
    t4t_fraction = np.sum(same_action)/n_samples,
    defect_fraction = np.sum([int(agent_action) for agent_action in agent_actions])/n_samples
    coop_fraction = 1 - defect_fraction
    
    return t4t_fraction[0], coop_fraction

def is_t4t_no_history(agent, n_samples, policy_id='default_policy'):
    
    agent_actions = []
    t4t_actions = []
    state = agent.get_policy(policy_id=policy_id).get_initial_state()
    
    for i in range(n_samples):
        history = generate_history_no_history()
        # action, state, _ = agent.compute_single_action(observation=history.flatten(), policy_id=policy_id)
        action, state, _ = agent.compute_single_action(observation=history.flatten(), state=state, policy_id=policy_id)

        opponents_last_action = int(np.where(history[1,:])[0])
            
        agent_actions.append(action)
        t4t_actions.append(opponents_last_action)
        
    same_action = [int(agent_actions[i]==t4t_actions[i]) for i in range(n_samples)]
    t4t_fraction = np.sum(same_action)/n_samples,
    defect_fraction = np.sum([int(agent_action) for agent_action in agent_actions])/n_samples
    coop_fraction = 1 - defect_fraction
    
    return t4t_fraction[0], coop_fraction
    

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
        