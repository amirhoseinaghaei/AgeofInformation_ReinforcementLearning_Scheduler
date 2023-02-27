import json
import numpy as np
from Rudder import LessonBuffer
from Environment import Environment
from Rudder import RRLSTM as LSTM
import torch
import time as Time
import random
from PolicyUpdater import PolicyUpdater
from matplotlib import pyplot as plt
from Dtos.StateDto import State
from OptimalPoliciyAgents.OptimalPolicyDDDQN import DDDQNOptimalPolicyAgent

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
environment = Environment(100,25)
environment.CreateStates()
policy_updator  = PolicyUpdater(environment= environment, lr = policy_lr)
Total_Reward_List = []
Total_Reward_List_Without_Optimal_Policy = []
# Opening JSON file
# with open('Optimal_Policy_DictDDDeepQN_0.2_70_125.json') as json_file:
#     Optimal_Policy_Dict = json.load(json_file)

OptimalPolicyAgent = DDDQNOptimalPolicyAgent()
Optimal_Policy_Dict = OptimalPolicyAgent.FindOptimalPolicy()

for i in range(NUM_EPISODE):
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
        action = 1
        if environment.state.Ra == 0 and environment.state.U == 0:
          action = 0
        if environment.state.Ra == 0 and environment.state.U == deadline - 1:
          action = 1
        if environment.state.U > 0:
          action = 0
        state, reward, done = environment.step(action)
        Episode_AoI.append(state[0])
        actions.append(action)
        states.append(state)
        rewards.append(reward) 
        if done: 
          Episode_AoI = [int(i) for i in Episode_AoI]
          Total_Reward_List_Without_Optimal_Policy.append(sum(Episode_AoI)/deadline)
    rewards = []
    states = [first_state]
    Episode_AoI = [first_state[0]]
    actions = []
    done = False
    environment.reset_paramter()
    environment.state = State(name, int(first_state[0]), (first_state[1]), int(first_state[2]), int(first_state[3]), int(first_state[4]))
    while not done:
        if UseOptimalPolicy:
          action  = 0 if Optimal_Policy_Dict[name] == 'wait' else 1 
        if environment.state.Ra == 0 and environment.state.U == 0:
          action = 0
        if environment.state.Ra == 0 and environment.state.U == 24:
          action = 1
        if environment.state.U > 0:
          action = 0
        state, reward, done = environment.step(action)

        Episode_AoI.append(state[0])
        actions.append(action)
        states.append(state)
        rewards.append(reward) 
        if done: 
          Episode_AoI = [int(i) for i in Episode_AoI]
          Total_Reward_List.append(sum(Episode_AoI)/deadline)


fig, ax = plt.subplots(2,figsize=(12,11))


AVerageAoIUsingScheduler = [sum(Total_Reward_List)/NUM_EPISODE for i in range(NUM_EPISODE)]
AVerageAoIWithoutScheduler = [sum(Total_Reward_List_Without_Optimal_Policy)/NUM_EPISODE for i in range(NUM_EPISODE)]

AoI_Difference_List = [Total_Reward_List_Without_Optimal_Policy[i] - Total_Reward_List[i] for i in range(len(Total_Reward_List_Without_Optimal_Policy)) ]

ax[1].plot(AoI_Difference_List, label ='Average AoI Difference', color = "b" ,mec = 'b', mfc = 'w', marker='.', markersize = 10, linestyle = "solid", linewidth = 3)

ax[0].plot(Total_Reward_List, label ='Using AoI Scheduler', color = "g" ,mec = 'g', mfc = 'w', marker='.', markersize = 20, linestyle = "solid", linewidth = 3)
ax[0].plot(Total_Reward_List_Without_Optimal_Policy, label ='Always sending back' , mec = 'r', mfc = 'w', color = "r", marker='.', markersize = 20,  linestyle = "solid", linewidth = 3)
ax[0].legend(fontsize = 20, loc='upper left')
ax[1].legend(fontsize = 20, loc='upper left')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys(), loc='upper left')

plt.show()