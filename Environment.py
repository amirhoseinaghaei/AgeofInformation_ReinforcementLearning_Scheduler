
import copy
import random
import time as Time
import numpy as np
from Dtos.StateDto import State
from WirelessChannel import DefiningWirelessChannels
import torch



class Environment(object):
    
    def __init__(self, B_max, C_max , WirelessChannelClass = DefiningWirelessChannels(2,[5,20]), K = 3 , W = 30 , Fc = 5):
        
        self.inital_state = None
        self.state = None
        self.updated = True
        self.update = None
        self.deadline = 25
        self.time = None
        self.Wait = None
        self.Fc = Fc
        self.K = K
        self.W = W
        self.B_max = B_max
        self.C_max = C_max
        self.WirelessChannelClass = WirelessChannelClass
        self.WirelessChannelClass.Create_RateDict()
        self.ConfigureParameters()
    
    def ConfigureParameters(self):
        self.BT_max = self.C_max + self.B_max
        self.Au_min = (self.B_max/max(self.WirelessChannelClass.Rate_List)) + (self.C_max/self.Fc)
        self.Au_max = self.W + self.W*self.K + (self.B_max/min(self.WirelessChannelClass.Rate_List)) + (self.C_max/self.Fc)
        self.ADT_max = (self.B_max/min(self.WirelessChannelClass.Rate_List)) + (self.C_max/self.Fc)

    def CreateStates(self):
        print("Amir")
        D = self.ADT_max - self.Au_min
        self.StateList = []
        for i in range(int(self.Au_min),int(self.Au_max)+1):
            if self.Au_min <= i <= self.W:
                BT = 0
                if self.Au_min <= i <= self.ADT_max + self.Au_min -1:
                    U_max = self.deadline 
                self.StateList.append(State(f"({i}, Ch1, {BT}, {1}, {0})", Au= i, Ch ="Ch1" , BT = BT, Ra = 1, U = 0))
                self.StateList.append(State(f"({i}, Ch2, {BT}, {1}, {0})", Au= i, Ch ="Ch2" , BT = BT, Ra = 1, U = 0))
                for j in range(0,int(U_max)+1):
                    self.StateList.append(State(f"({i}, Ch1, {BT}, {0}, {j})", Au= i, Ch ="Ch1" , BT = BT, Ra = 0, U = j))
                    self.StateList.append(State(f"({i}, Ch2, {BT}, {0}, {j})", Au= i, Ch ="Ch2" , BT = BT, Ra = 0, U = j))
            if i >= self.W and 0 <= i%self.W <= self.Au_min:
                first = self.BT_max - (i%self.W)*max(self.WirelessChannelClass.Rate_List)
                
                if first < self.C_max : 
                   
                    BT = []
                    Len = i%self.W- self.B_max/max(self.WirelessChannelClass.Rate_List)
                    Len = int(Len)
                    for j in range(Len+1):
                        BT.append(self.C_max-(j)*self.Fc)
                    BT.sort()
                   
                else:
                    BT = [first]
                last = self.BT_max - (i%self.W)*min(self.WirelessChannelClass.Rate_List)
                crowler = first
                while crowler != last:
                    BT.append(int(crowler + D)) if crowler + D > 25 else None
                    crowler = crowler + D
                if self.W <= i < self.deadline + self.Au_min :
                    U_max = self.deadline
                    for bt in BT:
                        self.StateList.append(State(f"({i}, Ch1, {bt}, {1}, {0})", Au= i, Ch ="Ch1" , BT = bt, Ra = 1, U = 0))
                        self.StateList.append(State(f"({i}, Ch2, {bt}, {1}, {0})", Au= i, Ch ="Ch2" , BT = bt, Ra = 1, U = 0))
                        for j in range(0,int(U_max)+1):
                            self.StateList.append(State(f"({i}, Ch1, {bt}, {0}, {j})", Au= i, Ch ="Ch1" , BT = bt, Ra = 0, U = j))
                            self.StateList.append(State(f"({i}, Ch2, {bt}, {0}, {j})", Au= i, Ch ="Ch2" , BT = bt, Ra = 0, U = j))
                else: 
                    U_max = self.deadline 
                    for bt in BT:
                        self.StateList.append(State(f"({i}, Ch1, {bt}, {1}, {0})", Au= i, Ch ="Ch1" , BT = bt, Ra = 1, U = 0))
                        self.StateList.append(State(f"({i}, Ch2, {bt}, {1}, {0})", Au= i, Ch ="Ch2" , BT = bt, Ra = 1, U = 0))
                        for j in range(0,int(U_max)+1):
                            self.StateList.append(State(f"({i}, Ch1, {bt}, {0}, {j})", Au= i, Ch ="Ch1" , BT = bt, Ra = 0, U = j))
                            self.StateList.append(State(f"({i}, Ch2, {bt}, {0}, {j})", Au= i, Ch ="Ch2" , BT = bt, Ra = 0, U = j))
            if i> self.C_max and i%self.W > self.Au_min and i%self.W < self.B_max/min(self.WirelessChannelClass.Rate_List):
                BT = [0,5,10,15,20,25]
                first = self.BT_max - (i%self.W)*max(self.WirelessChannelClass.Rate_List)
                if first <= 25 : 
                    BT = [0,5,10,15,20,25]
                else:
                    BT = [first]
                last = self.BT_max - (i%self.W)*min(self.WirelessChannelClass.Rate_List)
                crowler = first
                while crowler != last:
                    BT.append(int(crowler + D)) if crowler + D > 25 else None
                    crowler = crowler + D           
                U_max = self.deadline 
                for bt in BT:
                    self.StateList.append(State(f"({i}, Ch1, {bt}, {1}, {0})", Au= i, Ch ="Ch1" , BT = bt, Ra = 1, U = 0))
                    self.StateList.append(State(f"({i}, Ch2, {bt}, {1}, {0})", Au= i, Ch ="Ch2" , BT = bt, Ra = 1, U = 0))
                    for j in range(0,int(U_max)+1):
                        self.StateList.append(State(f"({i}, Ch1, {bt}, {0}, {j})", Au= i, Ch ="Ch1" , BT = bt, Ra = 0, U = j))
                        self.StateList.append(State(f"({i}, Ch2, {bt}, {0}, {j})", Au= i, Ch ="Ch2" , BT = bt, Ra = 0, U = j))
            if i> self.C_max and i%self.W > self.Au_min and i%self.W >= self.B_max/min(self.WirelessChannelClass.Rate_List):
                BT = [0]
                crowler = 0
                if int(self.C_max - (i%self.W - ((self.B_max)/min(self.WirelessChannelClass.Rate_List)))*self.Fc) > 0:
                    while crowler != int(self.C_max - (i%self.W - ((self.B_max)/min(self.WirelessChannelClass.Rate_List)))*self.Fc):
                        BT.append(crowler + self.Fc)  
                        crowler = crowler + self.Fc             
                U_max = self.deadline 
                for bt in BT:
                    self.StateList.append(State(f"({i}, Ch1, {bt}, {1}, {0})", Au= i, Ch ="Ch1" , BT = bt, Ra = 1, U = 0))
                    self.StateList.append(State(f"({i}, Ch2, {bt}, {1}, {0})", Au= i, Ch ="Ch2" , BT = bt, Ra = 1, U = 0))
                    for j in range(0,int(U_max)+1):
                        self.StateList.append(State(f"({i}, Ch1, {bt}, {0}, {j})", Au= i, Ch ="Ch1" , BT = bt, Ra = 0, U = j))
                        self.StateList.append(State(f"({i}, Ch2, {bt}, {0}, {j})", Au= i, Ch ="Ch2" , BT = bt, Ra = 0, U = j))

        self.initial_State  = []
        for i in self.StateList:
            if i.Ra == 1:

                if 70 <= i.BT <= 125: 
                    self.initial_State.append(i)   
        self.Quality = {}
        
        for i in self.StateList:
            self.Quality[(i.Name , 0)] = 0
            self.Quality[(i.Name , 1)] = 0  

    def reset_state_with_state(self, state):
        self.state = state
        self.inital_state = copy.deepcopy(self.state)
        return state
   
    def reset_state(self):
        self.state = copy.deepcopy(random.choice(self.initial_State))
        self.inital_state = copy.deepcopy(self.state)
        return np.array([self.state.Au, self.state.Ch, self.state.BT, self.state.Ra, self.state.U]), self.state

    def reset_paramter(self):
        self.Wait = 0
        self.time = 0
        self.updated = False
        self.Sendback = 0
    def remained_BT_modification(self):
        if self.state.Au % self.W < self.W-1 and self.state.BT == 0:
            self.state.BT = 0
        elif self.state.Au % self.W == self.W-1:
            self.state.BT = self.BT_max
        else:
            if self.C_max < self.state.BT <= self.BT_max:
                self.state.BT -= self.WirelessChannelClass.Rate_Dict[self.state.Ch]
            else:
                self.state.BT -= self.Fc
    def generate_channel_state_list_for_whole_sequence(self ,input_ch):
        self.ch_transition_list = [input_ch]
        for i in range(self.deadline + 1):
          random_generated = random.uniform(0, 1)
          if input_ch == "Ch1":
            if random_generated < self.WirelessChannelClass.TransitionProbabilityMatrix[0][0]:
                self.ch_transition_list.append("Ch1")
            else:
                self.ch_transition_list.append("Ch2")
          else:
            if random_generated < self.WirelessChannelClass.TransitionProbabilityMatrix[1][0]:
                self.ch_transition_list.append("Ch1")
            else:
                self.ch_transition_list.append("Ch2")
        return self.ch_transition_list

    def wireless_channel_modification(self, time):
      if time < self.deadline + 1:
        self.state.Ch = self.ch_transition_list[time]

    def request_pending_time_modification(self,action):
        # if action == 1:
        #     self.state.U = 0
        # elif action == 0 and self.Sendback:
        #     self.state.U = 0
        if self.updated == True or action == 1:
            self.state.U = 0
        else: 
            self.state.U += 1

    def AoI_modification(self, action):
        if self.Wait == True:  
            if self.update == False:
                self.state.Au += 1
            elif self.update and self.Sendback:
                self.state.Au += 1
            elif self.update and self.Sendback == 0:
                self.state.Au = self.state.Au % self.W + 1 
        else:
            self.updated = True
            if self.inital_state.BT == 0:
                self.state.Au = (self.state.Au%self.W) + 1
            else:
                self.state.Au = self.W + (self.state.Au%self.W) + 1
            self.Wait = True
    def state_transition(self, action):
            self.remained_BT_modification()
            self.wireless_channel_modification(self.time)
            self.request_pending_time_modification(action)
            self.state.Ra = 0
            self.AoI_modification(action)
            self.state.Name = f"({self.state.Au}, {self.state.Ch}, {self.state.BT}, {self.state.Ra}, {self.state.U})" 

    def step(self, action):
        reward = 0
        if self.time == 0:
            self.Wait = action == 0
            self.Sendback = action == 1
        self.time += 1
        done = self.time == self.deadline
        self.update = self.state.BT == self.Fc
        # if self.update and self.updated == False and self.Wait:
        #     # print("updated")
        #     self.updated = True
        #     if self.inital_state.Au >= self.W and self.inital_state.BT != 0:
        #             reward += self.W + (self.inital_state.Au)%(self.W)
        #     else:
        #         reward += (self.inital_state.Au)%(self.W)
        #     reward -= (self.time + (self.state.Au)%(self.W) + 1)
        if self.update and self.updated == False and self.Wait:
            self.updated = True
            if self.inital_state.Au >= self.W and self.inital_state.BT != 0:
              reward += (self.W + (self.inital_state.Au)%(self.W) + self.time - (self.state.Au + 1)%(self.W)) \
              *(self.deadline - (self.time))
              reward -= (self.inital_state.Au - (self.W + (self.inital_state.Au)%(self.W)))* self.time
            else:
              reward += ((self.inital_state.Au)%(self.W) + self.time - (self.state.Au)%(self.W) + 1) \
                    *(self.deadline - self.time)
              reward -= (self.inital_state.Au - ((self.inital_state.Au)%(self.W)))* self.time
        if self.update and self.updated == True and self.Sendback:
            if self.inital_state.Au >= self.W and self.inital_state.BT != 0:
              reward += (self.W + (self.inital_state.Au)%(self.W) + self.time - (self.state.Au + 1)%(self.W)) \
              *(self.deadline - (self.time))
              reward -= (self.inital_state.Au - (self.W + (self.inital_state.Au)%(self.W)))* self.time
    
            else:
              reward += ((self.inital_state.Au)%(self.W) + self.time - (self.state.Au)%(self.W) + 1) \
                    *(self.deadline - self.time)
              reward -= (self.inital_state.Au - ((self.inital_state.Au)%(self.W)))* self.time
            reward = -reward 

        # if self.update and self.updated == True and self.Sendback:
        #     if self.inital_state.Au >= self.W and self.inital_state.BT != 0:
        #             reward -= self.W + (self.inital_state.Au)%(self.W)
        #     else:
        #         reward -= (self.inital_state.Au)%(self.W)
        #     reward += (self.time + (self.state.Au)%(self.W) + 1)
        if done and self.Wait and self.updated == False:
              reward += ((self.inital_state.Au)%(self.W) + self.time - (self.state.Au)%(self.W) + 1) \
                    *(self.deadline - self.time)
              reward -= (self.inital_state.Au - ((self.inital_state.Au)%(self.W)))* self.time
        self.state_transition(action)
        # print(np.array([self.state.Au, self.state.Ch, self.state.BT, 0, self.state.U ]))
        # Time.sleep(2)
        return np.array([self.state.Au, self.state.Ch, self.state.BT, 0, self.state.U ]) , reward , done


            # Lesson_buffer.add(states = states, actions = actions, rewards = rewards)

