import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

from envs import MatrixGameEnv, MatrixGameEnv_no_history

from players import TitForTatPlayer, TitForTatThenDefectPlayer

import evaluation
import ray
from ray import tune
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.examples.models.shared_weights_model import \
    SharedWeightsModel1, SharedWeightsModel2, TF2SharedWeightsModel, \
    TorchSharedWeightsModel
from ray.rllib.models import ModelCatalog
# from ray.rllib.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG as DEFAULT_CONFIG_PPO

from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG 
from ray.rllib.agents.dqn import  DEFAULT_CONFIG as DEFAULT_CONFIG_DQN


from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

ray.init(ignore_reinit_error=True, log_to_driver=False)

register_env('MG_t4td_env', lambda c: MatrixGameEnv_no_history(
    player2=TitForTatThenDefectPlayer(min_defect_turn=0, max_defect_turn=100)))

register_env('MG_t4t_env', lambda c: MatrixGameEnv_no_history(
    player2=TitForTatPlayer()))


base_dir = '/home/peter/Documents/ML/rl_ipd/more_runs/MA_lstm/'
exp_dir = 'MA_PPO2/'
# exp_dir = 'PPO_lstm_single_t4td/'

cp_path = "/checkpoint_000100/checkpoint-100"
exps = os.listdir(base_dir+exp_dir)
test_exp = exps[0]


t4t_frac = []
coop_frac = []
for test_exp in exps:
    path1 = base_dir+ exp_dir+test_exp
# path1 = base_dir+ exp_dir+run_dir

# path1='/home/peter/Documents/ML/rl_ipd/single_agent_runs1/new_runs/DQN_single_t4t'
# path1='/home/peter/Documents/ML/rl_ipd/single_agent_runs1/new_runs/DQN_single_t4t/DQN_MG_t4t_env_3dc73_00000_0_gamma=0.999,lr=0.001,n_step=1_2021-08-13_15-33-36'
    
    if os.path.exists(path1 + cp_path):
        with open(path1 + '/params.pkl', 'rb') as f:
            data = pickle.load(f)
        agent = PPOTrainer(config=data)
        agent.restore(path1+ cp_path, )
        t_frac, c_frac = evaluation.is_t4t_no_history(agent,100)
        t4t_frac.append(t_frac[0])
        coop_frac.append(c_frac)