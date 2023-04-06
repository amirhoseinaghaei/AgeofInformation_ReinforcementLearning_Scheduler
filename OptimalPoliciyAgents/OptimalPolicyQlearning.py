import numpy as np
# from Rudder import LessonBuffer
from Environment import Environment
from WirelessChannel import DefiningWirelessChannels
from Config import SimulationParameters
from matplotlib import pyplot as plt
import time as Time
import torch
from tqdm import tqdm
import random
from PolicyUpdater import PolicyUpdater
from Dtos.StateDto import State
import json

class QlearningOptimalPolicy:
    def __init__(self, env, redistributer, policy_lr, Num_Of_Episodes) -> None:
        self.env = env
        self.reward_redistributer = redistributer
        self.policy_updator  = PolicyUpdater(environment= env, lr = policy_lr)
        self.Num_Of_Episodes = Num_Of_Episodes
    def generate_optimnal_policy(self):
       
        episode = 0
        for i in tqdm(range(self.Num_Of_Episodes)):
            self.env.reset_paramter()
            state , fixed_State = self.env.reset_state()
            first_state = state
            self.env.generate_channel_state_list_for_whole_sequence(state[1])
            episode += 1
            rewards = []
            states = [first_state]
            Episode_AoI = [first_state[0]]
            actions = []
            done = False
            name = f'({first_state[0]}, {first_state[1]}, {first_state[2]}, {first_state[3]}, {first_state[4]})'
            while not done:
                action = 1
                # if np.random.random() < 0.15:
                #     action = np.random.choice(2) 
                # else:
                #     action = 0 if self.policy_updator.Quality[name,0] > self.policy_updator.Quality[name,1] else 1
                if self.env.state.Ra == 0 and self.env.state.U == 0:
                    action = 0
                if self.env.state.U > 0:
                    action = 0
                if self.env.sendbackaction == True:
                    action = 1
                state, reward, done = self.env.step(action)
                Episode_AoI.append(state[0])
                actions.append(action)
                states.append(state)
                rewards.append(reward) 
                if done: 
                    for i in states: 
                        i[1] = i[1].split("h")[1]
                    states = np.stack(states)
                    states = states.astype(int)
                    rewards = np.array(rewards, dtype = np.float32)
                    actions = np.array(actions)
                    if actions[0] == 1: 
                        rewards = self.reward_redistributer.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
                    #     print(state)
                    #     print(rewards)
                    #     Time.sleep(5)
                    # else:
                    #     for i in range(len(rewards)):
                    #         rewards[i] = 0
                    self.policy_updator.Q_learning(actions= actions , states = states, rewards= rewards)

        Optimal_Policy_Dict = {}
        for keys, value in self.policy_updator.Quality.items():
                initial_StateName = []
                for i in self.env.initial_State:
                    initial_StateName.append(i.Name) 
                if keys[0] in initial_StateName: 
                    print(f"Waiting_Qvalue {keys[0]}: {self.policy_updator.Quality[keys[0],0]}")
                    print(f"Sending_Qvalue {keys[0]}: {self.policy_updator.Quality[keys[0],1]}")

                    if self.policy_updator.Quality[keys[0],0] > self.policy_updator.Quality[keys[0],1]:  
                        Optimal_Policy_Dict[keys[0]] = "wait"
                    else:
                        Optimal_Policy_Dict[keys[0]] = "send"
        with open(f"OptimalPolicies/Qlearning/Optimal_Policy_Qlearning_{self.env.B_max}.json", "w") as write_file:
            json.dump(Optimal_Policy_Dict, write_file, indent=4)

