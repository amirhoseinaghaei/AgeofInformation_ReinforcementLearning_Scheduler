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

lb_size = 2048
n_lstm = 16
max_time = 50
policy_lr = 0.5
lstm_lr = 1e-2
l2_regularization = 1e-6
avg_window = 750
UseOptimalPolicy = True
episode = 0
NUM_EPISODE = 100
deadline = 25
episode = 0
Training = False

simulationParameters = SimulationParameters('Configs.json')
simulationParameters.Configure()
WirelessChannels = DefiningWirelessChannels(simulationParameters.NumberOfTransmissionChannels)
environment = Environment(simulationParameters.NumberOfBits, simulationParameters.NumberOfCPUCycles, simulationParameters.IntervalBetweenArrivals,
                          simulationParameters.WindowSize, simulationParameters.CPUSpeed, WirelessChannels, simulationParameters.Deadline )
environment.CreateStates()
Lesson_buffer_a0 = LessonBuffer(100000, simulationParameters.Deadline, 5)
rudder_lstm_a0 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a0, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=500,
                        lstm_batch_size=32, continuous_pred_factor=0.5)
QlearningPolicyFinder = QlearningOptimalPolicy(policy_lr = policy_lr, redistributer= rudder_lstm_a0, env= environment, Num_Of_Episodes = NUM_EPISODE)

if Training == False:
  rudder_lstm_a0.load_state_dict(torch.load('rudder_lstm_1000_send.pt'))
else:
  QlearningPolicyFinder.generate_optimnal_policy()

Total_Reward_List = []
Total_Reward_List_With_Optimal_Policy = []
with open('OptimalPolicies/Qlearning/Optimal_Policy_Qlearning_1000.json') as json_file:
    Optimal_Policy_Dict_Qlearning = json.load(json_file)
for i in tqdm(range(NUM_EPISODE)):
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
        # if np.random.random() < 0.15:
        #     action = np.random.choice(2) 
        # else:
        #     action = 0 if policy_updator.Quality[name,0] > policy_updator.Quality[name,1] else 1
        action  = 0 if Optimal_Policy_Dict_Qlearning[name] == 'wait' else 1
        # action = 1
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
          # Total_Reward_List_Without_Optimal_Policy.append(sum(Episode_AoI)/deadline)
          Total_Reward_List_With_Optimal_Policy.append(sum(Episode_AoI)/deadline)
    rewards = []
    states = [first_state]
    Episode_AoI = [first_state[0]]
    actions = []
    done = False
    environment.reset_paramter()
    environment.state = State(name, int(first_state[0]), (first_state[1]), int(first_state[2]), int(first_state[3]), int(first_state[4]))
    while not done:
        # if UseOptimalPolicy:
        action  = 1
          # print(Optimal_Policy_Dict[name])
        # else:
        #   action = 1 
        if environment.state.Ra == 0 and environment.state.U == 0:
          action = 0
        # if environment.state.Ra == 0 and environment.state.U == 24:
        #   action = 1
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

          # for i in states: 
          #     i[1] = i[1].split("h")[1]
          # states = np.stack(states)
          # states = states.astype(int)
          # rewards = np.array(rewards, dtype = np.float32)
          # actions = np.array(actions)
          # if actions[0] == 1: 
          #     rewards = rudder_lstm_a0.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
              # for i in range(len(rewards)):
              #   rewards[i] = 0
              # print(rewards)
          # policy_updator.Q_learning(actions= actions , states = states, rewards= rewards)
          # for i in range(len(states)-1): 
          #   print(states[i])
          #   print(rewards[i])
          # Lesson_buffer_a0.add(states = states, actions = actions, rewards = rewards)
          # if  episode < 2500 and Lesson_buffer_a0.full_enough() :
          #         if episode % 25 == 0:
          #             print(episode)
          #             rudder_lstm_a0.train(episode=episode)
          #         if episode >= 500: 
          #             torch.save(rudder_lstm_a0.state_dict(), 'rudder_lstm_1000_send.pt')
# Optimal_Policy_Dict = {}
# for keys, value in policy_updator.Quality.items():
#          initial_StateName = []
#          for i in environment.initial_State:
#             initial_StateName.append(i.Name) 
#          if keys[0] in initial_StateName: 
#             if policy_updator.Quality[keys[0],0] > policy_updator.Quality[keys[0],1]:  
#               Optimal_Policy_Dict[keys[0]] = "wait"
#               print('{:15} {:15} {:15}'.format( keys[0] , "Wait" , policy_updator.Quality[keys[0] ,0]))
#             else:
#               Optimal_Policy_Dict[keys[0]] = "send"
#               print('{:15} {:15} {:15}'.format( keys[0] , "Send Back" , policy_updator.Quality[keys[0] ,1]))
# print(Optimal_Policy_Dict)
# with open("Optimal_Policy_Qlearning_1000.json", "w") as write_file:
#     json.dump(Optimal_Policy_Dict, write_file, indent=4)




          
    # print("*******************************************************************************")
    # for i in range(len(states)-1): 
    #   print(states[i])
    #   print(rewards[i])
fig, ax = plt.subplots(2,figsize=(12,11))
ax[0].plot(Total_Reward_List, label ='Respond immediately', mec = 'r', mfc = 'w', color = "r", marker='.', markersize = 18,  linestyle = "solid", linewidth = 3)
ax[0].plot(Total_Reward_List_With_Optimal_Policy, label ='Proposed response scheduler',  color = "g" ,mec = 'g', mfc = 'w', marker='*', markersize = 15, linestyle = "dashed", linewidth = 3)

plt.show()