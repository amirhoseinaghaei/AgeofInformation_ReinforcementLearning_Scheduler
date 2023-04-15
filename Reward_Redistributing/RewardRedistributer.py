import numpy as np
import torch


class Reward_Redistributer():
    def __init__(self,number_of_episodes, env, lstm_networks) -> None:
        self.number_of_episodes = number_of_episodes
        self.env = env
        self.lstm_networks = lstm_networks
    def generate_reward_redistributer(self, to_be_learned_action):
        print(f"To be learnt action: {to_be_learned_action}")
        episode = 0
        for i in range(self.number_of_episodes):

            self.env.reset_paramter()
            state , _ = self.env.reset_state()
            first_state = state
            self.env.generate_channel_state_list_for_whole_sequence(state[1])
            episode += 1
            rewards = []
            states = [first_state]
            actions = []
            done = False
            while not done:
                action = to_be_learned_action
                if self.env.state.Ra == 0 and self.env.state.U == 0:
                    action = 0
                if self.env.state.U > 0:
                    action = 0
                if self.env.sendbackaction == True:
                    action = 1
                state, reward, done = self.env.step(action)
                actions.append(action)
                states.append(state)
                rewards.append(reward) 
                if done: 
                    for i in states: 
                        i[1] = i[1].split("h")[1]
                    states = np.stack(states)
                    states = states.astype(float).astype(int)
                    rewards = np.array(rewards, dtype = np.float32)
                    actions = np.array(actions)
                    self.lstm_networks[to_be_learned_action].buffer.add(states = states, actions = actions, rewards = rewards)
                    if  episode < 5000  and self.lstm_networks[to_be_learned_action].buffer.full_enough() :
                            # and self.lstm_networks[to_be_learned_action].buffer.different_returns_encountered()/
                            if episode % 25 == 0:
                                print(episode)
                                self.lstm_networks[to_be_learned_action].train(episode=episode)
                            if episode >= 1800: 
                                torch.save(self.lstm_networks[to_be_learned_action].state_dict(), f'Reward_Redistributing/GeneratedRedistributers/rudder_lstm_{to_be_learned_action}_{self.env.B_max}.pt')
        return self.lstm_networks[to_be_learned_action]
