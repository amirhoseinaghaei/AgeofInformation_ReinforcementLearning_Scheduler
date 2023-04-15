import copy
import numpy as np
from DQNNetworks.DuelingDoubleDQN import Dueling_DDQN_Agent
# from Rudder import LessonBuffer
from Environment import Environment
from WirelessChannel.WirelessChannel import DefiningWirelessChannels
from Config import SimulationParameters
from matplotlib import pyplot as plt
import time as Time
import torch
from tqdm import tqdm
import random
from PolicyUpdater import PolicyUpdater
from Dtos.StateDto import State
import json

class DDDQNOptimalPolicy:
    def __init__(self, env, redistributers, Num_Of_Episodes) -> None:
        self.env = env
        self.reward_redistributers = redistributers
        self.DDDQN = Dueling_DDQN_Agent(gamma=1, epsilon=1.0, lr=3e-4,
                  input_dims=[5], n_actions=2, mem_size=100000, eps_min=0.01,
                  batch_size=64, eps_dec=1e-3, replace=100)
        self.Num_Of_Episodes = Num_Of_Episodes
    def generate_optimnal_policy(self):
       
        episode = 0
        for i in tqdm(range(5000)):
            self.env.reset_paramter()
            state , fixed_State = self.env.reset_state()
            first_state = copy.deepcopy(state)
            
            self.env.generate_channel_state_list_for_whole_sequence(state[1])
            episode += 1
            # first_state[1] = int(first_state[1].split("h")[1])
            # first_state = first_state.astype(float).astype(int)
            rewards = []
            states = [first_state]
            Episode_AoI = [first_state[0]]
            actions = []
            dones = []

            done = False
            name = f'({first_state[0]}, {first_state[1]}, {first_state[2]}, {first_state[3]}, {first_state[4]})'

            while not done:

                new_state = copy.deepcopy(state)
                action = self.DDDQN.choose_action(new_state)
                if self.env.state.Ra == 0 and self.env.state.U == 0:
                    action = 0
                if self.env.state.U > 0:
                    action = 0
                if self.env.sendbackaction == True:
                    action = 1
                state, reward, done = self.env.step(action)
                # print(state)
                Episode_AoI.append(state[0])
                actions.append(action)
                states.append(state)
                rewards.append(reward) 
                dones.append(done)

                if done: 
                    # states[0][1] = int(states[0][1].split("h")[1]) 
                    # states[0] = states[0].astype(float).astype(int)
                    # states[-1][1] = int(states[-1][1].split("h")[1]) 
                    # states[-1] = states[-1].astype(float).astype(int)
                    for i in states: 
                        i[1] = i[1].split("h")[1]
                    # for i in states: 
                    #     print(i)
                    #     i[1] = i[1].split("h")[1]
                    states = np.stack(states)
                    # print(states)

                    states = states.astype(float).astype(int)
                    rewards = np.array(rewards, dtype = np.float32)
                    actions = np.array(actions)
                    rewards = self.reward_redistributers[actions[0]].redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
           
                    for i in range(self.env.deadline):
                            self.DDDQN.store_transition(states[i], actions[i], rewards[i], states[i+1],
                                    dones[i])
                            self.DDDQN.learn()

        Optimal_Policy_Dict = {}
        # for keys, value in tqdm(self.policy_updator.Quality.items()):

        #         if self.policy_updator.Quality[keys[0],0] > self.policy_updator.Quality[keys[0],1]:  
        #             Optimal_Policy_Dict[keys[0]] = f"wait: {self.policy_updator.Quality[keys[0],0]}"
        #         else:
        #             Optimal_Policy_Dict[keys[0]] = f"send: {self.policy_updator.Quality[keys[0],1]}"
        # with open(f"OptimalPolicies/Qlearning/Optimal_Policy_Qlearning_{self.env.B_max}.json", "w") as write_file:
        #     json.dump(Optimal_Policy_Dict, write_file, indent=4)

        for state in self.env.initial_State:
            State  =  np.array([state.Au, state.Ch, state.BT, state.Ra, state.U])
            State[1] = State[1].split("h")[1]
            State = State.astype(float).astype(int) 
            state = torch.tensor([State],dtype=torch.float).to(self.DDDQN.q_eval.device)
            _, advantage =   self.DDDQN.q_eval.forward(state)
            action = torch.argmax(advantage).item()
            name = f'({State[0]}, {f"Ch{State[1]}"}, {State[2]}, {State[3]}, {State[4]})'
            if action == 0:
                Optimal_Policy_Dict[name] = f"wait: {advantage}"
            else:
                Optimal_Policy_Dict[name] = f"send: {advantage}"  
        with open(f"OptimalPolicies/DDDQN/Optimal_Policy_DDDQN_{self.env.B_max}.json", "w") as write_file:
            json.dump(Optimal_Policy_Dict, write_file, indent=4)

        return self.DDDQN
