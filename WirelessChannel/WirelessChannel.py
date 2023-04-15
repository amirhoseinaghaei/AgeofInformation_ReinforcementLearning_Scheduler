import numpy as np 
class DefiningWirelessChannels(object):
    def __init__(self, NUM_Ch):
        self.NUM_Ch = NUM_Ch
        self.Rate_Dict = {}
        self.Rate_List = []
        self.TransitionProbabilityMatrix = np.zeros((NUM_Ch,NUM_Ch))

    def Create_RateDict(self):
        for i in range(self.NUM_Ch):
            self.Rate_Dict[f"Ch{i+1}"] = ((i*0.5)+2)*10
            self.Rate_List.append(((i*0.5)+2)*10)

        for x in range(self.NUM_Ch):
            for y in range(self.NUM_Ch):
                self.TransitionProbabilityMatrix[x][y] = np.random.uniform(0,1)
                self.TransitionProbabilityMatrix[x] = self.TransitionProbabilityMatrix[x]/sum(self.TransitionProbabilityMatrix[x])        
        print(self.TransitionProbabilityMatrix)
        print(self.Rate_Dict)

