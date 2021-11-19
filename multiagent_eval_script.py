import numpy as np
import matplotlib.pyplot as plt


import os
# import cloudpickle as pickle
import pickle5 as pickle
# import joblib as pickle
import re
import envs
from envs import MatrixGameEnv, MatrixGameEnv_no_history

import players
from players import TitForTatPlayer, TitForTatThenDefectPlayer

import evaluation


import pandas as pd

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

register_env('two_agent_MG_env', lambda c: envs.TwoAgentMatrixGameEnv())

register_env('two_agent_t4t_MG_env', lambda c: envs.TwoAgentSeparateMatrixGameEnv())

register_env('MG_t4tTD_env', lambda c: envs.MatrixGameEnv(player2=players.TitForTatThenDefect()))
# register_env('MG_t4t_env', lambda c: envs.MatrixGameEnv(player2=players.TitForTat()))
register_env('MG_t4t_env', lambda c: envs.MatrixGameEnv())


base_dir = '/home/peter/Documents/ML/rl_ipd/MA_runs/big1/MA/'
exp_dirs = ['MA_DQN1/']
env_pref = 'DQN'

# exp_dir = 'DQN_single_t4td/'
# env_pref = 'DQN_MG_t4td_env'

cp_path = "/checkpoint_000200/checkpoint-200"
# exps = os.listdir(base_dir+exp_dir)

# base_dir = '/home/peter/Documents/ML/rl_ipd/MA_runs/small1/MA/'
# exp_dirs = ['MA_DQN2/']
# env_pref = 'DQN'

# # exp_dir = 'DQN_single_t4td/'
# # env_pref = 'DQN_MG_t4td_env'

# cp_path = "/checkpoint_000100/checkpoint-100"
# # exps = os.listdir(base_dir+exp_dir)

base_dir = '/home/peter/Documents/ML/rl_ipd/more_runs/'
exp_dirs = ['MA_random_length_DQN2/', 'MA_random_length_nodoneatend_DQN2/', 'MA_random_length_nodoneatend_DQN2_2/']
env_pref = 'DQN'
cp_path = "/checkpoint_000100/checkpoint-100"
state_len=None

# base_dir = '/home/peter/Documents/ML/rl_ipd/more_runs/'
# exp_dirs = ['MA_random_length_DQN3/']
# env_pref = 'DQN'
# cp_path = "/checkpoint_000400/checkpoint-400"



# base_dir = '/home/peter/Documents/ML/rl_ipd/more_runs/MA_statelen/'
# exp_dirs = ['MA_random_length_nodoneatend_DQN1_statelen1/','MA_random_length_nodoneatend_DQN2_statelen1/',
#             'MA_random_length_DQN1_statelen1/', 'MA_random_length_DQN2_statelen1/',
#             ]
# state_len=1

# exp_dirs = ['MA_random_length_nodoneatend_DQN1_statelen5/','MA_random_length_nodoneatend_DQN2_statelen5/',
#             'MA_random_length_DQN1_statelen5/', 'MA_random_length_DQN2_statelen5/']
# state_len=5

env_pref = 'DQN'
cp_path = "/checkpoint_000100/checkpoint-100"

for exp_dir in exp_dirs:
    exps = os.listdir(base_dir+exp_dir)
    attributes = ['gamma', 'lr']
    data_names = ['episode_reward_max', 'episode_reward_min', 'episode_reward_mean']
    data1 = pd.DataFrame(columns=['ID']+attributes+data_names)

    
    re_dict = {}
    for attr in attributes:
        p = re.compile(attr + '=[^_^,]*')
        re_dict[attr] = p

    data_names = ['episode_reward_max', 'episode_reward_min', 'episode_reward_mean']
    data1 = pd.DataFrame(columns=['ID']+attributes+data_names)

    for test_exp in exps:

        path1 = base_dir+ exp_dir+test_exp

        if os.path.isdir(path1) and (env_pref in test_exp):
            print('hon')

            append_dict = {}
            append_dict['ID'] = test_exp
            has_none = False
            for attr in attributes:
                p = re_dict[attr]
                val_str = p.findall(test_exp)
                if not val_str:
                    val = None
    #                 print(at)
                    has_none = True
                    break

                else:
                    val = val_str[0][len(attr)+1:]
                    val = float(val)
                append_dict[attr] = val
                if os.path.exists(path1 + cp_path):
                    print('hi')
                    progress_csv = pd.read_csv(path1+'/progress.csv')
                    
                    
                    for data_name in data_names:
                        vals = progress_csv[data_name].to_numpy()
                        append_dict[data_name] = vals
                        append_dict['final_' + data_name] = vals[-1]
#                     try:
                    with open(path1 + '/params.pkl', 'rb') as f:
                        data = pickle.load(f)
                    # print(data)
                    # assert False
                    agent = DQNTrainer(config=data)
                    agent.restore(path1+ cp_path, )
                    t_frac, c_frac = evaluation.is_t4t(agent,100, policy_id='agent-0', state_len=state_len)

                    append_dict['t4t_frac0'] = t_frac
                    append_dict['coop_frac0'] = c_frac

                    t_frac, c_frac = evaluation.is_t4t(agent,100, policy_id='agent-1', state_len=state_len)

                    append_dict['t4t_frac1'] = t_frac
                    append_dict['coop_frac1'] = c_frac
#                     except:
#                         print('nope')


                else:
                    has_none = True
                    break

            if not has_none:
                data1 = data1.append(append_dict,ignore_index=True)
    #             print(append_dict)
    print(data1)
    data1.to_pickle(base_dir + exp_dir + 'data_save')