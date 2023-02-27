import sys
sys.path.insert(0, '..')
import numpy as np
from Rudder import LessonBuffer
from Environment import Environment
from Rudder import RRLSTM as LSTM
import torch
import time as Time
import random
from PolicyUpdater import PolicyUpdater
from tqdm import tqdm
import json
from DQNAgents.DuelingDoubleDQN import Dueling_DDQN_Agent
import torch as T


class DDDQNOptimalPolicyAgent(object):
    def __init__(self):
        self.n_lstm = 16
        self.lstm_lr = 1e-2
        self.l2_regularization = 1e-6
        self.rudder_lstm_a0 = LSTM(state_input_size=5, n_actions= 2, buffer= LessonBuffer(1000, 20, 5), n_units=self.n_lstm,
                        lstm_lr=self.lstm_lr, l2_regularization=self.l2_regularization, return_scaling=10,
                        lstm_batch_size=8, continuous_pred_factor=0.5)
        self.rudder_lstm_a1 = LSTM(state_input_size=5, n_actions= 2, buffer=LessonBuffer(1000, 20, 5), n_units=self.n_lstm,
                        lstm_lr=self.lstm_lr, l2_regularization=self.l2_regularization, return_scaling=10,
                        lstm_batch_size=8, continuous_pred_factor=0.5)

        self.rudder_lstm_a0.load_state_dict(torch.load('LSTM_Networks/rudder_lstm_70_125_wait_0.2.pt'))
        self.rudder_lstm_a1.load_state_dict(torch.load('LSTM_Networks/rudder_lstm_70_125_send_0.2.pt'))
        self.Dueling_Double_DQN_Agent = Dueling_DDQN_Agent(gamma=1, epsilon=1.0, lr=3e-4,
                        input_dims=[5], n_actions=2, mem_size=100000, eps_min=0.01,
                        batch_size=64, eps_dec=1e-3, replace=100)
        self.environment = Environment(100,25)
        self.environment.CreateStates()
    def FindOptimalPolicy(self):
        episode = 0
        for i in tqdm(range(5000)):
            episode += 1
            self.environment.reset_paramter()
            state, _ = self.environment.reset_state()
            self.environment.generate_channel_state_list_for_whole_sequence(state[1])
            rewards = []
            states = [state]
            actions = []
            dones = []
            done = False
            name = f'({state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]})'

            initial_state= state
            if initial_state[1] == "Ch1":
                initial_state[1] = 1
            else:
                initial_state[1] = 0
            initial_state = initial_state.astype(int)

            while not done:
                action = self.Dueling_Double_DQN_Agent.choose_action(initial_state)
                if self.environment.state.Ra == 0 and self.environment.state.U == 0:
                    action = 0
                if self.environment.state.Ra == 0 and self.environment.state.U == 19:
                    action = 1
                if self.environment.state.U > 0:
                    action = 0
                state, reward, done = self.environment.step(action)
                actions.append(action)
                states.append(state)
                rewards.append(reward) 
                dones.append(done)
                if done: 
                    res = np.nonzero(rewards)[0]
                    if len(res) > 0 :
                        rewards[-1] = rewards[res[0]]
                        rewards[res[0]] = 0   
                    for i in states: 
                        if i[1] == "Ch1":
                            i[1] = 1
                        else:
                            i[1] = 0
                    states = np.stack(states)
                    states = states.astype(int)
                    rewards = np.array(rewards, dtype = np.float32)
                    actions = np.array(actions)
                    if actions[0] == 0: 
                        rewards = self.rudder_lstm_a0.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
                    if actions[0] == 1: 
                        rewards = self.rudder_lstm_a1.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
                    for i in range(20):
                        self.Dueling_Double_DQN_Agent.store_transition(states[i], actions[i], rewards[i], states[i+1],
                                            dones[i])
                        self.Dueling_Double_DQN_Agent.learn()
        Optimal_Policy_Dict = {}
        for state in self.environment.initial_State:
            State  =  np.array([state.Au, state.Ch, state.BT, state.Ra, state.U])
            if State[1] == "Ch1":
                State[1] = 1
            else:
                State[1] = 0
            State = State.astype(int) 
            state = T.tensor([State],dtype=T.float).to(self.Dueling_Double_DQN_Agent.q_eval.device)
            _, advantage =  self.Dueling_Double_DQN_Agent.q_eval.forward(state)
            action = T.argmax(advantage).item()
            if State[1] == 1:
                CH_NAME = "Ch1"
            else:
                CH_NAME = "Ch2"
            name = f'({State[0]}, {CH_NAME}, {State[2]}, {State[3]}, {State[4]})'
            if action == 0:
                Optimal_Policy_Dict[name] = "wait"
            else:
                Optimal_Policy_Dict[name] = "send"  
        with open("Optimal_Policy_DictQ_learning_0.2_70_125.json", "w") as write_file:
            json.dump(Optimal_Policy_Dict, write_file, indent=4)  

        return Optimal_Policy_Dict


