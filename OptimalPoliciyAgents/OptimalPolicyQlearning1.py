# import sys
# sys.path.insert(0, '..')
# import numpy as np
# from Rudder import LessonBuffer
# from Environment import Environment
# from Rudder import RRLSTM as LSTM
# import torch
# import time as Time
# import random
# from PolicyUpdater import PolicyUpdater
# import json


# lb_size = 2048
# n_lstm = 16
# max_time = 50
# policy_lr = 0.5
# lstm_lr = 1e-2
# l2_regularization = 1e-6
# avg_window = 750

# episode = 0
# Lesson_buffer_a1 = LessonBuffer(1000, 25, 5)
# Lesson_buffer_a0 = LessonBuffer(1000, 25, 5)

# rudder_lstm_a0 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a0, n_units=n_lstm,
#                         lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10,
#                         lstm_batch_size=8, continuous_pred_factor=0.5)
# rudder_lstm_a1 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a1, n_units=n_lstm,
#                         lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10,
#                         lstm_batch_size=8, continuous_pred_factor=0.5)
# import os
# print("........." + os.getcwd())
# rudder_lstm_a0.load_state_dict(torch.load('../LSTM_Networks/rudder_lstm_70_125_send_0.7.pt'))
# rudder_lstm_a1.load_state_dict(torch.load('../LSTM_Networks/rudder_lstm_70_125_send_0.7.pt'))

# environment = Environment(100,25)
# environment.CreateStataes()
# policy_updator  = PolicyUpdater(environment= environment, lr = policy_lr)
# episode = 0
# for i in range(5000):
#     episode += 1
#     environment.reset_paramter()
#     state , _ = environment.reset_state()
#     environment.generate_channel_state_list_for_whole_sequence(state[1])
#     rewards = []
#     states = [state]
#     actions = []
#     done = False
#     name = f'({state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]})'

#     while not done:
#         if np.random.random() < 0.15:
#             action = np.random.choice(2) 
#         else:
#             action = 0 if policy_updator.Quality[name,0] > policy_updator.Quality[name,1] else 1
#         if environment.state.Ra == 0 and environment.state.U == 0:
#             action = 0
#         if environment.state.Ra == 0 and environment.state.U == 24:
#             action = 1
#         if environment.state.U > 0:
#             action = 0
#         state, reward, done = environment.step(action)
    
#         actions.append(action)
#         states.append(state)
#         rewards.append(reward) 
#         if done: 

#             res = np.nonzero(rewards)[0]
#             if len(res) > 0 :
#               # print(res)
#               rewards[-1] = rewards[res[0]]
#               rewards[res[0]] = 0   
#             for i in states: 
#                 if i[1] == "Ch1":
#                     i[1] = 1
#                 else:
#                     i[1] = 0
#             states = np.stack(states)
#             states = states.astype(int)
#             rewards = np.array(rewards, dtype = np.float32)
#             actions = np.array(actions)
#             # if actions[0] == 0: 
#             rewards = rudder_lstm_a0.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
#             # if actions[0] == 1: 
#             #   rewards = rudder_lstm_a1.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
            
#             policy_updator.Q_learning(actions= actions , states = states, rewards= rewards)

# Optimal_Policy_Dict = {}
# for keys, value in policy_updator.Quality.items():
#          initial_StateName = []
#          for i in environment.initial_State:
#             initial_StateName.append(i.Name) 
#          if keys[0] in initial_StateName: 
#             if policy_updator.Quality[keys[0],0] > policy_updator.Quality[keys[0],1]:  
#               print('{:15} {:15} {:15}'.format( keys[0] , "Wait" , policy_updator.Quality[keys[0] ,0]))

#               Optimal_Policy_Dict[keys[0]] = "wait"
#             else:
#               Optimal_Policy_Dict[keys[0]] = "send"
#               print('{:15} {:15} {:15}'.format( keys[0] , "Send Back" , policy_updator.Quality[keys[0] ,1]))

# with open("Optimal_Policy_DictQ_learning_0.7_70_125.json", "w") as write_file:
#     json.dump(Optimal_Policy_Dict, write_file, indent=4)

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
import json
from tqdm import tqdm

lb_size = 2048
n_lstm = 16
max_time = 50
policy_lr = 0.5
lstm_lr = 1e-2
l2_regularization = 1e-6
avg_window = 750
deadline = 50
Lesson_buffer_a0 = LessonBuffer(1000, deadline, 5)
episode = 0
rudder_lstm_a0 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a0, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=500,
                        lstm_batch_size=32, continuous_pred_factor=0.5)
# rudder_lstm_a0.load_state_dict(torch.load('rudder_lstm_120_send_0.1.pt.pt'))
environment = Environment(1000,100)
environment.CreateStates()
policy_updator  = PolicyUpdater(environment= environment, lr = policy_lr)
episode = 0
pos_rewards = 0
neg_rewards = 0
for i in tqdm(range(5000)):
    episode += 1
    environment.reset_paramter()
    state, _ = environment.reset_state()
    environment.generate_channel_state_list_for_whole_sequence(state[1])
    rewards = []
    states = [state]
    actions = []
    done = False
    name = f'({state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]})'
# if policy_updator.Quality[name,0] > policy_updator.Quality[name,1] else 1
    while not done:
        action = 1
        if environment.state.Ra == 0 and environment.state.U == 0:
            action = 0
        if environment.state.U > 0:
            action = 0
        if environment.sendbackaction == True:
            action = 1

        state, reward, done = environment.step(action)
        actions.append(action)
        states.append(state)
        rewards.append(reward) 
        if done: 

            res = np.nonzero(rewards)[0]
            if rewards[-1] == 0 and len(res) > 0 :
              rewards[-1] = sum(rewards)
              rewards[-1] = rewards[-1]
              rewards[res[0]] = 0
            else:
              rewards[-1] = rewards[-1]
            if rewards[-1] > 0:
              pos_rewards += 1
            if rewards[-1] < 0:
              neg_rewards += 1
            for i in states: 
                if i[1] == "Ch1":
                    i[1] = 1
                else:
                    i[1] = 0
            states = np.stack(states)
            states = states.astype(int)
            rewards = np.array(rewards, dtype = np.float32)
            actions = np.array(actions)
            # print(states)
            # print(rewards)
            Lesson_buffer_a0.add(states = states, actions = actions, rewards = rewards)
            if  episode < 2500 and Lesson_buffer_a0.full_enough() and Lesson_buffer_a0.different_returns_encountered()  :
                    if episode % 25 == 0:

                        print(episode)
                        rudder_lstm_a0.train(episode=episode)
                    if episode >= 1800: 
                        torch.save(rudder_lstm_a0.state_dict(), 'rudder_lstm_500_send_0.2.pt')

            rewards = rudder_lstm_a0.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
            policy_updator.Q_learning(actions= actions , states = states, rewards= rewards)

Optimal_Policy_Dict = {}
for keys, value in policy_updator.Quality.items():
         initial_StateName = []
         for i in environment.initial_State:
            initial_StateName.append(i.Name) 
          
         if keys[0] in initial_StateName: 
            print('{:15} {:15} {:15}'.format( keys[0] , keys[1] , policy_updator.Quality[keys[0] ,keys[1]]))
            if policy_updator.Quality[keys[0],0] > policy_updator.Quality[keys[0],1]:  
              Optimal_Policy_Dict[keys[0]] = "wait"
              print('{:15} {:15} {:15}'.format( keys[0] , "Wait" , policy_updator.Quality[keys[0] ,0]))
            else:
              Optimal_Policy_Dict[keys[0]] = "send"
              print('{:15} {:15} {:15}'.format( keys[0] , "Send Back" , policy_updator.Quality[keys[0] ,1]))
print(Optimal_Policy_Dict)
print(pos_rewards)
print(neg_rewards)