import copy
import numpy as np
from Environment import Environment
from OptimalPoliciyAgents.OptimalPolicyDDDQN import DDDQNOptimalPolicy
from WirelessChannel.WirelessChannel import DefiningWirelessChannels
from Config import SimulationParameters
from matplotlib import pyplot as plt
from RUDDER.LSTM import RRLSTM as LSTM
from RUDDER.ReplayBuffer import LessonBuffer
import time as Time
import torch
from tqdm import tqdm
from Dtos.StateDto import State
from OptimalPoliciyAgents.OptimalPolicyQlearning import QlearningOptimalPolicy
import json
from Reward_Redistributing.RewardRedistributer import Reward_Redistributer
import ujson

lb_size = 2048
n_lstm = 64
policy_lr = 0.5
lstm_lr = 1e-3
l2_regularization = 1e-6
episode = 0
NUM_EPISODE = 5000
Training = True
OptimalPolicyTraining = True
simulationParameters = SimulationParameters('Configs.json')
simulationParameters.Configure()
WirelessChannels = DefiningWirelessChannels(simulationParameters.NumberOfTransmissionChannels)
environment = Environment(simulationParameters.NumberOfBits, simulationParameters.NumberOfCPUCycles, simulationParameters.IntervalBetweenArrivals,
                          simulationParameters.WindowSize, simulationParameters.CPUSpeed, WirelessChannels, simulationParameters.Deadline )
environment.CreateStates()
Lesson_buffer_a0 = LessonBuffer(100000, simulationParameters.Deadline, 5)
rudder_lstm_a0 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a0, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10000,
                        lstm_batch_size=32, continuous_pred_factor=0.5)
Lesson_buffer_a1 = LessonBuffer(100000, simulationParameters.Deadline, 5)
rudder_lstm_a1 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a0, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10000,
                        lstm_batch_size=32, continuous_pred_factor=0.5)

RewardRedistributerGenerator = Reward_Redistributer(NUM_EPISODE, environment, [rudder_lstm_a0,rudder_lstm_a1])
if OptimalPolicyTraining:
  if Training == False:
    rudder_lstm_a0.load_state_dict(torch.load(f'Reward_Redistributing/GeneratedRedistributers/rudder_lstm_0_{environment.B_max}.pt'))
    rudder_lstm_a1.load_state_dict(torch.load(f'Reward_Redistributing/GeneratedRedistributers/rudder_lstm_1_{environment.B_max}.pt'))
    DDDQNPolicyFinder = DDDQNOptimalPolicy(env= environment , Num_Of_Episodes= NUM_EPISODE, redistributers= [rudder_lstm_a0, rudder_lstm_a1])
    DDDQN_Network = DDDQNPolicyFinder.generate_optimnal_policy()
    QlearningPolicyFinder = QlearningOptimalPolicy(policy_lr = policy_lr, redistributers= [rudder_lstm_a0, rudder_lstm_a1], env= environment, Num_Of_Episodes = NUM_EPISODE)
    qulaity_dict = QlearningPolicyFinder.generate_optimnal_policy()

  else:
    rudder_lstm_a0 = RewardRedistributerGenerator.generate_reward_redistributer(0)
    rudder_lstm_a1 = RewardRedistributerGenerator.generate_reward_redistributer(1)
    DDDQNPolicyFinder = DDDQNOptimalPolicy(env= environment , Num_Of_Episodes= NUM_EPISODE, redistributers= [rudder_lstm_a0, rudder_lstm_a1])
    DDDQN_Network = DDDQNPolicyFinder.generate_optimnal_policy()
    QlearningPolicyFinder = QlearningOptimalPolicy(policy_lr = policy_lr, redistributers= [rudder_lstm_a0, rudder_lstm_a1], env= environment, Num_Of_Episodes = NUM_EPISODE)
    qulaity_dict = QlearningPolicyFinder.generate_optimnal_policy()


# else: 
#    with open(f'OptimalPolicies/Qlearning/Optimal_Policy_Qlearning_{environment.B_max}.json') as json_file:
#     qulaity_dict = json.load(json_file)
Total_Reward_List_Qlearing = []
Total_Reward_List_With_DDDQN = []


for i in tqdm(range(1000)):
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
        # action  = 0 if qulaity_dict[name , 0] >= qulaity_dict[name , 1] else 1
        new_state = copy.deepcopy(state)
        action = DDDQN_Network.choose_action(new_state)
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
          Total_Reward_List_With_DDDQN.append(sum(Episode_AoI)/simulationParameters.Deadline)
    rewards = []
    states = [first_state]
    Episode_AoI = [first_state[0]]
    actions = []
    done = False
    environment.reset_paramter()
    environment.state = State(name, int(float(first_state[0])), (first_state[1]), int(float(first_state[2])), int(first_state[3]), int(first_state[4]))
    while not done:
        # action  = 1
        action  = 0 if qulaity_dict[name , 0] >= qulaity_dict[name , 1] else 1
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
          Total_Reward_List_Qlearing.append(sum(Episode_AoI)/simulationParameters.Deadline)

fig, ax = plt.subplots(2,figsize=(12,11))
ax[0].plot(Total_Reward_List_Qlearing, label ='Q-learning Proposed response scheduler', mec = 'r', mfc = 'w', color = "r", marker='.', markersize = 18,  linestyle = "solid", linewidth = 3)
ax[0].plot(Total_Reward_List_With_DDDQN, label ='DDDQN Proposed response scheduler',  color = "g" ,mec = 'g', mfc = 'w', marker='*', markersize = 15, linestyle = "dashed", linewidth = 3)

plt.show()