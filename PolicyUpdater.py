import numpy as np          
class PolicyUpdater(object):
    
    def __init__(self, environment, lr):
        self.Quality = environment.Quality
        self.lr = lr
    def Q_estimation(self, states, actions, rewards):
 
        for i in range(actions.shape[0]):      
            ch_name = "Ch1" if states[i][1] == 1 else "Ch2"
            name = f'({states[i][0]}, {ch_name}, {states[i][2]}, {states[i][3]}, {states[i][4]})'
            self.Quality[name,actions[i]] += self.lr*(rewards[i] + rewards[i] - self.Quality[name,actions[i]])
            # print(f"Quallity: {self.Quality[name,actions[i]]}")
        return self.Quality
    def Q_learning(self, states, actions, rewards):
        gamma = 1
        for i in range(actions.shape[0]):      
            ch_name = "Ch1" if states[i][1] == 1 else "Ch2"
            name = f'({states[i][0]}, {ch_name}, {states[i][2]}, {states[i][3]}, {states[i][4]})'
            ch_name_next = "Ch1" if states[i+1][1] == 1 else "Ch2"
            name_next = f'({states[i+1][0]}, {ch_name_next}, {states[i+1][2]}, {states[i+1][3]}, {states[i+1][4]})'
            if (name, actions[i]) not in self.Quality.keys() :
              self.Quality[name, actions[i]]
            if (name_next, 1) not in self.Quality.keys():
              self.Quality[name_next, 1] = 0
            if (name_next, 0) not in self.Quality.keys():
              self.Quality[name_next, 0] = 0
            if i == 0:
              self.Quality[name,actions[i]] += self.lr*(rewards[i] + gamma*max(self.Quality[name_next,0],self.Quality[name_next,1]) - self.Quality[name,actions[i]])
            else:
              self.Quality[name,actions[i]] += self.lr*(rewards[i] + gamma * self.Quality[name_next,0] - self.Quality[name,actions[i]])
            # print(f"Quallity: {self.Quality[name,actions[i]]}")
        return self.Quality
