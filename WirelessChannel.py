class DefiningWirelessChannels(object):
    def __init__(self, NUM_Ch , Rate_List):
        self.NUM_Ch = NUM_Ch
        self.Rate_List = Rate_List
        self.Rate_Dict = {}
    def Create_RateDict(self):
        for i in range(self.NUM_Ch):
            self.Rate_Dict[f"Ch{i+1}"] = self.Rate_List[i]

        self.TransitionProbabilityMatrix = []
        for x in range(self.NUM_Ch):
            row = []
            for y in range(self.NUM_Ch):
                row.append(0.0 + x/5 + y*(1 - 2*(0.0 + x/5)))
            self.TransitionProbabilityMatrix.append(row)
        print(self.TransitionProbabilityMatrix)
      
WirelessChannel = DefiningWirelessChannels(2,[5,20])        
WirelessChannel.Create_RateDict()   

