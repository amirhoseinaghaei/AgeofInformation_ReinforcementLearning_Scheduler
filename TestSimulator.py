import numpy as np
# from Rudder import LessonBuffer
from Environment import Environment
from WirelessChannel.WirelessChannel import DefiningWirelessChannels
from Config import SimulationParameters
from matplotlib import pyplot as plt
from RUDDER.LSTM import RRLSTM as LSTM
from RUDDER.ReplayBuffer import LessonBuffer
import time as Time
import torch
from tqdm import tqdm
import random
from PolicyUpdater import PolicyUpdater
from Dtos.StateDto import State
from OptimalPoliciyAgents.OptimalPolicyQlearning import QlearningOptimalPolicy
import json
from Reward_Redistributing.RewardRedistributer import Reward_Redistributer
lb_size = 2048
n_lstm = 64
max_time = 50
policy_lr = 0.5
lstm_lr = 1e-3
l2_regularization = 1e-6
avg_window = 750
UseOptimalPolicy = True
episode = 0
NUM_EPISODE = 5000
deadline = 25
episode = 0
Training = False
OptimalPolicyTraining = True
simulationParameters = SimulationParameters('Configs.json')
simulationParameters.Configure()
WirelessChannels = DefiningWirelessChannels(simulationParameters.NumberOfTransmissionChannels)
environment = Environment(simulationParameters.NumberOfBits, simulationParameters.NumberOfCPUCycles, simulationParameters.IntervalBetweenArrivals,
                          simulationParameters.WindowSize, simulationParameters.CPUSpeed, WirelessChannels, simulationParameters.Deadline )
environment.CreateStates()
Lesson_buffer_a0 = LessonBuffer(100000, simulationParameters.Deadline, 5)
rudder_lstm_a0 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a0, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=1000,
                        lstm_batch_size=32, continuous_pred_factor=0.5)
Lesson_buffer_a1 = LessonBuffer(100000, simulationParameters.Deadline, 5)
rudder_lstm_a1 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a0, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=1000,
                        lstm_batch_size=32, continuous_pred_factor=0.5)

RewardRedistributerGenerator = Reward_Redistributer(NUM_EPISODE, environment, [rudder_lstm_a0,rudder_lstm_a1])
if OptimalPolicyTraining:
  if Training == False:
    rudder_lstm_a0.load_state_dict(torch.load('Reward_Redistributing/GeneratedRedistributers/rudder_lstm_0_1000.pt'))
    rudder_lstm_a1.load_state_dict(torch.load('Reward_Redistributing/GeneratedRedistributers/rudder_lstm_1_1000.pt'))
    QlearningPolicyFinder = QlearningOptimalPolicy(policy_lr = policy_lr, redistributers= [rudder_lstm_a0, rudder_lstm_a1], env= environment, Num_Of_Episodes = NUM_EPISODE)
    QlearningPolicyFinder.generate_optimnal_policy()

  else:
    rudder_lstm_a0 = RewardRedistributerGenerator.generate_reward_redistributer(0)
    rudder_lstm_a1 = RewardRedistributerGenerator.generate_reward_redistributer(1)
    QlearningPolicyFinder = QlearningOptimalPolicy(policy_lr = policy_lr, redistributers= [rudder_lstm_a0, rudder_lstm_a1], env= environment, Num_Of_Episodes = NUM_EPISODE)
    QlearningPolicyFinder.generate_optimnal_policy()
Total_Reward_List = []
Total_Reward_List_With_Optimal_Policy = []
with open('OptimalPolicies/Qlearning/Optimal_Policy_Qlearning_1000.json') as json_file:
    Optimal_Policy_Dict_Qlearning = json.load(json_file)
for i in tqdm(range(100)):
    environment.reset_paramter()
    state , fixed_State = environment.reset_state()
    first_state = state
    environment.generate_channel_state_list_for_whole_sequence(state[1])
    episode += 1
    rewards = []
    states = [first_state]
    Episode_AoI = [first_state[0]]
    actions = []
    done = False
    name = f'({first_state[0]}, {first_state[1]}, {first_state[2]}, {first_state[3]}, {first_state[4]})'

    while not done:
        action  = 0 if Optimal_Policy_Dict_Qlearning[name] == 'wait' else 1
        print(action)
        if environment.state.Ra == 0 and environment.state.U == 0:
          action = 0
        if environment.state.U > 0:
          action = 0
        if environment.sendbackaction == True:
          action = 1
        state, reward, done = environment.step(action)
        Episode_AoI.append(state[0])
        actions.append(action)
        states.append(state)
        rewards.append(reward) 
        if done: 
          Episode_AoI = [int(i) for i in Episode_AoI]
          Total_Reward_List_With_Optimal_Policy.append(sum(Episode_AoI)/deadline)
    rewards = []
    states = [first_state]
    Episode_AoI = [first_state[0]]
    actions = []
    done = False
    environment.reset_paramter()
    environment.state = State(name, int(first_state[0]), (first_state[1]), int(first_state[2]), int(first_state[3]), int(first_state[4]))
    while not done:
        action  = 1
        if environment.state.Ra == 0 and environment.state.U == 0:
          action = 0
        if environment.state.U > 0:
          action = 0
        if environment.sendbackaction == True:
          action = 1
        state, reward, done = environment.step(action)

        Episode_AoI.append(state[0])
        actions.append(action)
        states.append(state)
        rewards.append(reward) 
        if done: 
          Episode_AoI = [int(i) for i in Episode_AoI]
          Total_Reward_List.append(sum(Episode_AoI)/deadline)

fig, ax = plt.subplots(2,figsize=(12,11))
ax[0].plot(Total_Reward_List, label ='Respond immediately', mec = 'r', mfc = 'w', color = "r", marker='.', markersize = 18,  linestyle = "solid", linewidth = 3)
ax[0].plot(Total_Reward_List_With_Optimal_Policy, label ='Proposed response scheduler',  color = "g" ,mec = 'g', mfc = 'w', marker='*', markersize = 15, linestyle = "dashed", linewidth = 3)

plt.show()