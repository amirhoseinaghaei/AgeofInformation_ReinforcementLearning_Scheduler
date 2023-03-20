
import numpy as np
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
policy_lr = 0.5
lstm_lr = 1e-2
l2_regularization = 1e-6
avg_window = 750
deadline = 25
Lesson_buffer_a0 = LessonBuffer(1000, deadline, 5)
episode = 0
rudder_lstm_a0 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a0, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10,
                        lstm_batch_size=8, continuous_pred_factor=0.5)
# rudder_lstm_a0.load_state_dict(torch.load('LSTM_Networks/rudder_lstm_70_125_send_0.7.pt'))
environment = Environment(100,25)
environment.CreateStates()
policy_updator  = PolicyUpdater(environment= environment, lr = policy_lr)
episode = 0
for i in range(2500):
    episode += 1
    environment.reset_paramter()
    state, _ = environment.reset_state()
    environment.generate_channel_state_list_for_whole_sequence(state[1])
    rewards = []
    states = [state]
    actions = []
    done = False
    name = f'({state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]})'
    while not done:
        action = 1
        if environment.state.Ra == 0 and environment.state.U == 0:
            action = 0
        # if environment.state.Ra == 0 and environment.state.U == environment.time:
        #     action = 1
        if environment.state.U > 0:
            action = 0
        # if environment.sendnackaction == True:
        #     action = 1
        state, reward, done = environment.step(action)
        actions.append(action)
        states.append(state)
        rewards.append(reward) 
        if done: 

            res = np.nonzero(rewards)[0]
            if len(res) > 0 :
              rewards[-1] = rewards[res[0]]
              rewards[-1] = rewards[-1]/10
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
            Lesson_buffer_a0.add(states = states, actions = actions, rewards = rewards)
            if  episode < 5000 and Lesson_buffer_a0.full_enough() and Lesson_buffer_a0.different_returns_encountered()  :
                    if episode % 25 == 0:
                        print(episode)
                        rudder_lstm_a0.train(episode=episode)
                    if episode >= 1800: 
                        torch.save(rudder_lstm_a0.state_dict(), 'LSTM_Networks/rudder_lstm_70_125_send_0.7.pt')

            rewards = rudder_lstm_a0.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
    

