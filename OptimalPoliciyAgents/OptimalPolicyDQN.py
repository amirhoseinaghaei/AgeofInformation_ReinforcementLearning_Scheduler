import numpy as np
from Rudder import LessonBuffer
from Environment import Environment
from Rudder import RRLSTM as LSTM
import torch
import time as Time
import random
from PolicyUpdater import PolicyUpdater
from tqdm import tqdm
lb_size = 2048
n_lstm = 16
max_time = 50
policy_lr = 0.5
lstm_lr = 1e-2
l2_regularization = 1e-6
avg_window = 750

episode = 0

Lesson_buffer_a1 = LessonBuffer(1000, 25, 5)
Lesson_buffer_a0 = LessonBuffer(1000, 25, 5)

rudder_lstm_a0 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a0, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10,
                        lstm_batch_size=8, continuous_pred_factor=0.5)
rudder_lstm_a1 = LSTM(state_input_size=5, n_actions= 2, buffer=Lesson_buffer_a1, n_units=n_lstm,
                        lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10,
                        lstm_batch_size=8, continuous_pred_factor=0.5)
rudder_lstm_a0.load_state_dict(torch.load('rudder_lstm_70_125_wait_0.2.pt'))
rudder_lstm_a1.load_state_dict(torch.load('rudder_lstm_70_125_send_0.2.pt'))


DQN_Agent = Agent(gamma=1, epsilon=1.0, batch_size=64, n_actions=2, eps_end=0.01,
                  input_dims=[5], lr=0.001)

environment = Environment(100,25)
environment.CreateStates()
policy_updator  = PolicyUpdater(environment= environment, lr = policy_lr)
episode = 0
for i in tqdm(range(8000)):
    episode += 1
    environment.reset_paramter()
    state, _ = environment.reset_state()
    environment.generate_channel_state_list_for_whole_sequence(state[1])

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
   
        action = DQN_Agent.choose_action(initial_state)
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
              rewards = rudder_lstm_a0.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
            if actions[0] == 1: 
              rewards = rudder_lstm_a1.redistribute_reward(states=np.expand_dims(states, 0),actions=np.expand_dims(actions, 0))[0, :]
            for i in range(25):
              DQN_Agent.store_transition(states[i], actions[i], rewards[i], states[i+1],
                                  dones[i])
              DQN_Agent.learn()
Optimal_Policy_Dict = {}
for state in environment.initial_State:
  State  =  np.array([state.Au, state.Ch, state.BT, state.Ra, state.U])
  if State[1] == "Ch1":
    State[1] = 1
  else:
    State[1] = 0
  State = State.astype(int) 
  actions =  DQN_Agent.Q_eval.forward(State)

  if State[1] == 1:
    CH_NAME = "Ch1"
  else:
    CH_NAME = "Ch2"
  name = f'({State[0]}, {CH_NAME}, {State[2]}, {State[3]}, {State[4]})'
  action = T.argmax(actions).item()
  if action == 0:
    Optimal_Policy_Dict[name] = "wait"
  else:
    Optimal_Policy_Dict[name] = "send"  
with open("Optimal_Policy_DictQ_learning_0.2_70_125.json", "w") as write_file:
    json.dump(Optimal_Policy_Dict, write_file, indent=4)