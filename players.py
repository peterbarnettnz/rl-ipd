import numpy as np

class TitForTatPlayer():
    """Starts by cooperating, and then always plays the opponents previous move
    
    """
    
    def __init__(self):
        
        self.count = 0
        self.opponents_move = 0
        
    def play(self):
        
        return self.opponents_move
    
    def update(self, opponents_move):
        self.opponents_move = opponents_move
        
    def reset(self):
        
        self.count = 0
        self.opponents_move = 0
 

class TitForTatThenDefectFixedPlayer():
    """Starts playing tit-for-tat then starts defecting, at a specific turn
    
    """
    
    def __init__(self, n_defect_turn=50):
        
        self.count = 0
        self.opponents_move = 0
        self.n_defect_turn = n_defect_turn
        
    def play(self):
        
        if self.count < self.n_defect_turn:
            return self.opponents_move
        else:
            return 1
    
    def update(self, opponents_move):
        
        self.opponents_move = opponents_move
        self.count += 1
        
        
    def reset(self):
        
        self.count = 0
        self.opponents_move = 0
 
class TitForTatOrRandomPlayer():
    """Each episode either plays tit-for-tat or randomly
    
    """
    
    def __init__(self):
        
        self.count = 0
        self.opponents_move = 0
        self.is_random = np.random.randint(2, dtype=np.bool)
        
    def play(self):
        
        if self.is_random:
            return np.random.randint(2)
        else:
            return self.opponents_move
    
    def update(self, opponents_move):
        
        self.opponents_move = opponents_move
        self.count += 1
        
        
    def reset(self):
        
        self.count = 0
        self.opponents_move = 0
        self.is_random = np.random.randint(2, dtype=np.bool)

 
class TitForTatOrDefectPlayer():
    """Each episode either plays tit-for-tat or always-defect
    
    """
    
    def __init__(self):
        
        self.count = 0
        self.opponents_move = 0
        self.is_random = np.random.randint(2, dtype=np.bool)
        
    def play(self):
        
        if self.is_random:
            return 1
        else:
            return self.opponents_move
    
    def update(self, opponents_move):
        
        self.opponents_move = opponents_move
        self.count += 1
        
        
    def reset(self):
        
        self.count = 0
        self.opponents_move = 0
        self.is_random = np.random.randint(2, dtype=np.bool)

class TitForTatThenDefectPlayer():
    """Starts playing tit-for-tat, then starts always defecting at a random turn
    
    """
    
    def __init__(self, min_defect_turn = 50, max_defect_turn=100):
        
        self.count = 0
        self.opponents_move = 0
        self.is_random = np.random.randint(2, dtype=np.bool)
        self.min_defect_turn = min_defect_turn
        self.max_defect_turn = max_defect_turn
        
        self.defect_turn = np.random.randint(low=self.min_defect_turn, high=self.max_defect_turn)
        
    def play(self):
        
        if self.count > self.defect_turn:
            return 1
        else:
            return self.opponents_move
    
    def update(self, opponents_move):
        
        self.opponents_move = opponents_move
        self.count += 1
        
        
    def reset(self):
        
        self.count = 0
        self.opponents_move = 0
        self.is_random = np.random.randint(2, dtype=np.bool)
        self.defect_turn = np.random.randint(low=self.min_defect_turn, high=self.max_defect_turn)

 
class GrudgePlayer():
    """Starts by cooperating, but defects forever if opponent defects
    
    """
    
    def __init__(self):
        
        self.count = 0
#         self.opponents_move = 0
        self.grudging = False # has the opponent played defect
        
    def play(self):
        
        if self.grudging:
            return 1
        
        else:
            return 0
    
    def update(self, opponents_move):
#         self.opponents_move = opponents_move
        if opponents_move == 1:
            self.grudging = True
        
    def reset(self):
        
        self.count = 0
        self.grudging = False  


class RandomPlayer():
    """Cooperates a with a fixed probability
    
    """
    
    def __init__(self, P_coop=0.5):
        
        self.count = 0
        self.P_coop = P_coop
        
        assert self.P_coop <= 1.
    
        
    def play(self):
        
        if np.random.rand() < self.P_coop:
            return 0 
        
        else:
            return 1
        
    
    def update(self, opponents_move):
        pass
        
    def reset(self):
        
        self.count = 0


class RandomProbPlayer():
    """Cooperates a with a probability P_coop,
    P_coop is chosen randomly each round
    
    """
    
    def __init__(self):
        
        self.count = 0
        self.P_coop = np.random.rand()
            
        
    def play(self):
        
        if np.random.rand() < self.P_coop:
            return 0 
        
        else:
            return 1
        
    
    def update(self, opponents_move):
        pass
        
    def reset(self):
        
        self.count = 0
        self.P_coop = np.random.rand()


class CooperateOrDefectPlayer():
    """Either always cooperates or always defects each round
    
    """
    
    def __init__(self):
        
        self.count = 0
        self.P_coop = np.random.randint(2)
            
        
    def play(self):
        
        if np.random.rand() < self.P_coop:
            return 0 
        
        else:
            return 1
        
    
    def update(self, opponents_move):
        pass
        
    def reset(self):
        
        self.count = 0
        self.P_coop = np.random.randint(2)

class CooperatePlayer(RandomPlayer):
    """ Always cooperates
    """
    def __init__(self):
        super().__init__(P_coop=1.)

class DefectPlayer(RandomPlayer):
    """Always defects
    """
     
    def __init__(self):
        super().__init__(P_coop=0.)

