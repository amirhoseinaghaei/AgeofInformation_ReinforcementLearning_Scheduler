


import numpy as np
# from Environment import Actor
from Rudder import LessonBuffer
from Environment import Environment
from Rudder import RRLSTM as LSTM
import torch
import time as Time
import random
from PolicyUpdater import PolicyUpdater

lb_size = 2048
n_lstm = 16
max_time = 50
policy_lr = 0.1
lstm_lr = 1e-2
l2_regularization = 1e-6
avg_window = 750

Lesson_buffer = LessonBuffer(1000, 25, 5)
episode = 0
rudder_lstm = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10,
                        lstm_batch_size=8, continuous_pred_factor=0.5)

# rudder_lstm.load_state_dict(torch.load('rudder_lstm.pt'))
environment = Environment(100,25)
environment.CreateStates()
# print(type(environment.StateList[0].Name))
policy_updator  = PolicyUpdater(environment= environment, lr = policy_lr)
episode = 0
visited_dict = {}
for i in range(2000):
    episode += 1
    environment.reset_paramter()
    state = environment.reset_state()
    rewards = []
    states = [state]
    actions = []
    done = False
    name = f'({state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]})'

    while not done:
        if np.random.random() < 0.05:
            action = np.random.choice(2)
            if len(states) == 1 and action == 0:
              if name not in visited_dict.keys(): 
                  visited_dict[name] = 1
              else:
                  visited_dict[name] += 1
    
        else:
            action = 0 if policy_updator.Quality[name,0] > policy_updator.Quality[name,1] else 1          
            if len(states) == 1 and action == 0:
              if name not in visited_dict.keys(): 
                  visited_dict[name] = 1
              else:
                  visited_dict[name] += 1
        if environment.state.Ra == 0 and environment.state.U == 0:
            action = 0
        if environment.state.Ra == 0 and environment.state.U == 24:
            action = 1
        if environment.state.U > 0:
            action = 0
        state, reward, done = environment.step(action)
    
        actions.append(action)
        states.append(state)
        rewards.append(reward) 
        if done: 

            res = np.nonzero(rewards)[0]
            if len(res) > 0 :
              # print(res)
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
            Lesson_buffer.add(states = states, actions = actions, rewards = rewards)
            if  Lesson_buffer.full_enough() and Lesson_buffer.different_returns_encountered()  :
                    # print("different_returns_encountered")        
                    # If RUDDER is run, the LSTM is trained after each episode until its loss is below a threshold.
                    # Samples will be drawn from the lessons buffer.
                    if episode % 25 == 0:

                        # print("True")
                        print(episode)
                        rudder_lstm.train(episode=episode)
                    if episode >= 1600: 
                        torch.save(rudder_lstm.state_dict(), 'rudder_lstm.pt')
                    # Then the LSTM is used to redistribute the reward.
            # print(rewards)
            # print(states)
            # print(actions)
            rewards = rudder_lstm.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
            policy_updator.Q_estimation(actions= actions , states = states, rewards= rewards)
            # print(rewards)

for keys, value in policy_updator.Quality.items():
         initial_StateName = []
         for i in environment.initial_State:
            initial_StateName.append(i.Name) 
         if keys[0] in initial_StateName and keys[1] == 0: 
            print('{:15} {:15} {:15}'.format( keys[0] ,  keys[1], value))
for value in visited_dict:
        print('{:15} {:15}'.format( value, visited_dict[value]))
