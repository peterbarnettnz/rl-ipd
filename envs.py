import numpy as np
import gym
from gym import spaces
import players

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MatrixGame():
    
    def __init__(self, RPST=(3,1,0,5)):
        
        self.RPST = RPST
        
        self.payoff_mat = np.empty((2,2), dtype=np.object)
        
        self.payoff_mat[0, 0] = (RPST[0], RPST[0])
        self.payoff_mat[1, 1] = (RPST[1], RPST[1])
        self.payoff_mat[0, 1] = (RPST[2], RPST[3])
        self.payoff_mat[1, 0] = (RPST[3], RPST[2])
        
    def play(self, a_row, a_col):
        # for ease of things 0 is coooperate
        #                    1 is defect

        return self.payoff_mat[a_row, a_col]
        
class MatrixGameStackEnv(gym.Env):
    """ Environment for a matrix game with fixed history size. The history 'stacks'/fills up
        such that the moves on turn n are always stored at position n.

        (note that this just used to be called MatrixGameEnv)
    
    """
    
    def __init__(self,  RPST=(3,1,0,5), history_n=100, player2=players.TitForTatPlayer()):
        
        self.RPST = RPST
        self.history_n = history_n
        self.history = np.zeros((2,2,self.history_n))
        self._counter = 0
        self._setup_spaces()
        self.player2 = player2
        self.game = MatrixGame(RPST=(3,1,0,5))
        
        
        
    def _setup_spaces(self):
        
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(0, 1,
                                           shape=(self.history_n * 4,))
        
    def history_to_state(self, history=None):
        
        if history is None:
            history = self.history
            
        state = history.flatten()
        
        return state

    def step(self, action):
        
        assert self.action_space.contains(action)
        
        action2 = self.player2.play()
        rewards = self.game.play(action, action2)
        
        self.player2.update(action)
        self.history[0, action, self._counter] = 1
        self.history[1, action2, self._counter] = 1
        
        self.state = self.history_to_state(self.history)
        
        self._counter += 1
        done = self._counter >= self.history_n 
        
        return self.state, rewards[0], done, {}

        
        
    def reset(self):

        self.history = np.zeros((2,2,self.history_n))
        self.state = self.history.flatten()

        self._counter = 0
        self.player2.reset()
        
        return self.state

        
class MatrixGameRollingHistoryEnv(gym.Env):
    
    def __init__(self,  RPST=(3,1,0,5), history_n=10, game_length=100,player2=players.TitForTatPlayer()):
        
        self.RPST = RPST
        self.history_n = history_n
        self.game_length = game_length
        self.history = np.zeros((2,2,self.history_n))
        self._counter = 0
        self._setup_spaces()
        self.player2 = player2
        self.game = MatrixGame(RPST=RPST)
        
        
        
    def _setup_spaces(self):
        
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(0, 1,
                                           shape=(self.history_n * 4,))
        
    def history_to_state(self, history=None):
        
        if history is None:
            history = self.history
            
        state = history.flatten()
        
        return state
    
    def update_history(self, action1, action2):
        
#         if self._counter < self.history_n:
#             self.history[0, action, self._counter] = 1
#             self.history[1, action2, self._counter] = 1
#         else:
        self.history[:,:,:-1] = self.history[:,:,1:]
        self.history[:,:,-1] = 0
        self.history[0, action1, -1] = 1
        self.history[1, action2, -1] = 1
        

    def step(self, action):
        
        assert self.action_space.contains(action)
        
        action2 = self.player2.play()
        rewards = self.game.play(action, action2)
        
        self.player2.update(action)
        

        
        self.update_history(action, action2)
        
        self.state = self.history_to_state(self.history)
        self._counter += 1
        
        done = self._counter >= self.game_length

        return self.state, rewards[0], done, {}

        
        
    def reset(self):

        self.history = np.zeros((2,2,self.history_n))
        self.state = self.history.flatten()

        self._counter = 0
        self.player2.reset()
        
        return self.state

        
class MatrixGameEnv(gym.Env):
    
    def __init__(self,  RPST=(3,1,0,5), history_n=100, player2=players.TitForTatPlayer()):
        
        self.RPST = RPST
        self.history_n = history_n
        self.history = np.zeros((2,2,self.history_n))
        self._counter = 0
        self._setup_spaces()
        self.player2 = player2
        self.game = MatrixGame(RPST=RPST)
        
        
    def _setup_spaces(self):
        
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(0, 1,
                                           shape=(self.history_n * 4,))
        
    def history_to_state(self, history=None):
        
        if history is None:
            history = self.history
            
        state = history.flatten()
        
        return state
    
    def update_history(self, action1, action2):
        
#         if self._counter < self.history_n:
#             self.history[0, action, self._counter] = 1
#             self.history[1, action2, self._counter] = 1
#         else:
        self.history[:,:,1:] = self.history[:,:,:-1]
        self.history[:,:,0] = 0
        self.history[0, action1, 0] = 1
        self.history[1, action2, 0] = 1
        

    def step(self, action):
        
        assert self.action_space.contains(action)
        
        action2 = self.player2.play()
        rewards = self.game.play(action, action2)
        
        self.player2.update(action)
        self.update_history(action, action2)
        
        self.state = self.history_to_state(self.history)
        self._counter += 1
        
        done = self._counter >= self.history_n

        return self.state, rewards[0], done, {}

        
        
    def reset(self):

        self.history = np.zeros((2,2,self.history_n))
        self.state = self.history.flatten()

        self._counter = 0
        self.player2.reset()
        
        return self.state


class MatrixGameEnv_2player(gym.Env):
    
    def __init__(self,  RPST=(3,1,0,5), history_n=100):
        
        self.RPST = RPST
        self.history_n = history_n
        self.history = np.zeros((2,2,self.history_n))
        self._counter = 0
        self._setup_spaces()
        self.game = MatrixGame(RPST=RPST)
        
        
        
    def _setup_spaces(self):
        
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(0, 1,
                                           shape=(self.history_n * 4,))
        
    def history_to_state(self, history=None):
        
        if history is None:
            history = self.history
            
        state = history.flatten()
        
        return state
    
    def update_history(self, action1, action2):
        
#         if self._counter < self.history_n:
#             self.history[0, action, self._counter] = 1
#             self.history[1, action2, self._counter] = 1
#         else:
        self.history[:,:,1:] = self.history[:,:,:-1]
        self.history[:,:,0] = 0
        self.history[0, action1, 0] = 1
        self.history[1, action2, 0] = 1
        

    def step(self, action, action2):
        
        assert self.action_space.contains(action)
        
#         action2 = self.player2.play()
        rewards = self.game.play(action, action2)
                

        
        self.update_history(action, action2)
        
        self.state = self.history_to_state(self.history)
        self._counter += 1
        
        done = self._counter >= self.history_n

        return self.state, rewards[0], done, {}

    def reset(self):

        self.history = np.zeros((2,2,self.history_n))
        self.state = self.history.flatten()

        self._counter = 0
        
        return self.state

        
        
class MatrixGameRollingHistoryEnv_2player(gym.Env):
    
    def __init__(self,  RPST=(3,1,0,5), history_n=10, game_length=100):
        
        self.RPST = RPST
        self.history_n = history_n
        self.game_length = game_length
        self.history = np.zeros((2,2,self.history_n))
        self._counter = 0
        self._setup_spaces()
        self.game = MatrixGame(RPST=RPST)
        
        
        
    def _setup_spaces(self):
        
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(0, 1,
                                           shape=(self.history_n * 4,))
        
    def history_to_state(self, history=None):
        
        if history is None:
            history = self.history
            
        state = history.flatten()
        
        return state
    
    def update_history(self, action1, action2):
        
#         if self._counter < self.history_n:
#             self.history[0, action, self._counter] = 1
#             self.history[1, action2, self._counter] = 1
#         else:
        self.history[:,:,:-1] = self.history[:,:,1:]
        self.history[:,:,-1] = 0
        self.history[0, action1, -1] = 1
        self.history[1, action2, -1] = 1
        

    def step(self, action, action2):
        
        assert self.action_space.contains(action)
        
#         action2 = self.player2.play()
        rewards = self.game.play(action, action2)
                

        
        self.update_history(action, action2)
        
        self.state = self.history_to_state(self.history)
        self._counter += 1
        
        done = self._counter >= self.game_length

        return self.state, rewards[0], done, {}

        
        
    def reset(self):

        self.history = np.zeros((2,2,self.history_n))
        self.state = self.history.flatten()

        self._counter = 0
        
        return self.state

        
class MatrixGameEnv_no_history(gym.Env):
    
    def __init__(self,  RPST=(3,1,0,5), n_games=100, player2=players.TitForTatPlayer()):
        
        self.RPST = RPST
        self.last_moves = np.zeros((2,2))
        self._counter = 0
        self._setup_spaces()
#         self.player2 = GrudgePlayer()
#         self.player2 = DefectPlayer()
#         self.player2 = CooperateOrDefectPlayer()
    #         self.player2 = TitForTatThenDefectPlayer()
#         self.player2 = TitForTatOrRandom()
        self.player2 = player2
        self.n_games = n_games
        self.game = MatrixGame(RPST=(3,1,0,5))
        
        
        
    def _setup_spaces(self):
        
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(0, 1,
                                           shape=(4,))
        
    def lastmoves_to_state(self, last_moves=None):
        
        if last_moves is None:
            last_moves = self.last_moves
            
        state = last_moves.flatten()
        
        return state

    def step(self, action):
        
        assert self.action_space.contains(action)
        
        action2 = self.player2.play()
        rewards = self.game.play(action, action2)
        
        self.player2.update(action)
        self.last_moves = np.zeros((2,2))

        self.last_moves[0, action] = 1
        self.last_moves[1, action2] = 1
        
        self.state = self.lastmoves_to_state(self.last_moves)
        
        self._counter += 1
        done = self._counter >= self.n_games
        
        return self.state, rewards[0], done, {}

        
        
    def reset(self):

        self.last_moves = np.zeros((2,2))
        self.state = self.last_moves.flatten()

        self._counter = 0
        self.player2.reset()
        
        return self.state

        
        
class TwoAgentMatrixGameEnv(MultiAgentEnv):
    
    def __init__(self, RPST=(3,1,0,5), history_n=100):
        
        self.num_agents = 2
        
        self.RPST = RPST
        self.history_n = history_n
        self.history = np.zeros((2,2,self.history_n))
        
        self._counter = 0
        self._setup_spaces()
        self.game = MatrixGame(RPST=self.RPST)
    
    
    def _setup_spaces(self):
        
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(0, 1,
                                           shape=(self.history_n * 4,))
        
        
    def history_to_states(self, history=None):
        
        if history is None:
            history = self.history
            
        state1 = history.flatten()
        state2 = history[::-1,:,:].flatten()
        
        states = {0: state1, 1:state2}
        
        return states
            
    def update_history(self, action_dict):
        
#         if self._counter < self.history_n:
#             self.history[0, action, self._counter] = 1
#             self.history[1, action2, self._counter] = 1
#         else:
        self.history[:,:,1:] = self.history[:,:,:-1]
        self.history[:,:,0] = 0
        self.history[0, action_dict[0], 0] = 1
        self.history[1, action_dict[1], 0] = 1

    def step(self, action_dict):
        
        rewards = self.game.play(action_dict[0], action_dict[1])
        rew = {i: rewards[i] for i in [0, 1]}

        self.update_history(action_dict)
        
        obs = self.history_to_states(self.history)
        
        self._counter += 1
        
        is_done = self._counter >= self.history_n
        done = {i: is_done for i in [0, 1, "__all__"]}
        
        info = {0:{}, 1:{}}
        
        return obs, rew, done, info
        
        
        
    def reset(self):
        
        self.history = np.zeros((2,2,self.history_n))
        obs = self.history_to_states(self.history)

        self._counter = 0
        
        return obs
        