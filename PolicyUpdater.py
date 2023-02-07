import numpy as np          
class PolicyUpdater(object):
    
    def __init__(self, environment, lr):
        self.Quality = environment.Quality
        self.lr = lr
    def Q_estimation(self, states, actions, rewards):
 
        for i in range(actions.shape[0]):      
            ch_name = "Ch1" if states[i][1] == 1 else "Ch2"
            name = f'({states[i][0]}, {ch_name}, {states[i][2]}, {states[i][3]}, {states[i][4]})'
            self.Quality[name,actions[i]] += self.lr*(rewards[i] - self.Quality[name,actions[i]])
            # print(f"Quallity: {self.Quality[name,actions[i]]}")
        return self.Quality
