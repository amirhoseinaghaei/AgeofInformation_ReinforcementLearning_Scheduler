import numpy as np 
class DefiningWirelessChannels(object):
    def __init__(self, NUM_Ch):
        self.NUM_Ch = NUM_Ch
        self.Rate_Dict = {}
        self.Rate_List = []
        self.TransitionProbabilityMatrix = np.zeros((NUM_Ch,NUM_Ch))

    def Create_RateDict(self):
        for i in range(self.NUM_Ch):
            self.Rate_Dict[f"Ch{i+1}"] = ((i*2)+1)*20
            self.Rate_List.append((i+1)*20)

        for x in range(self.NUM_Ch):
            for y in range(self.NUM_Ch):
                # row.append(0.9 + x/10  + y*(1 - 2*(0.9 + x/10)))
                self.TransitionProbabilityMatrix[x][y] = 0.5
        print(self.TransitionProbabilityMatrix)

