import time
import numpy as np 


class LessonBuffer:
    def __init__(self, size, deadline, state_variables):
        self.size = size 
        self.states_buffer = np.empty(shape=(size, deadline + 1, state_variables))
        self.actions_buffer = np.empty(shape=(size, deadline))
        self.rewards_buffer = np.empty(shape=(size, deadline))
        self.lens_buffer = np.empty(shape=(size, 1), dtype=np.int32)
        self.next_spot_to_add = 0
        self.buffer_is_full = False
        self.samples_since_last_training = 0       
    def different_returns_encountered(self):
        if self.buffer_is_full:
            return np.unique(self.rewards_buffer[..., -1]).shape[0] > 1
        else:
            return np.unique(self.rewards_buffer[:self.next_spot_to_add, -1]).shape[0] > 1
    def full_enough(self):
        return self.buffer_is_full or self.next_spot_to_add > 50  
    def add(self,states, actions, rewards):
        traj_length = states.shape[0]
        # print(traj_length)
        next_ind = self.next_spot_to_add
        self.next_spot_to_add = self.next_spot_to_add + 1
        if self.next_spot_to_add >= self.size:
            self.buffer_is_full = True
        self.next_spot_to_add = self.next_spot_to_add % self.size
        # print(states)
        # print(states.squeeze())
        self.states_buffer[next_ind, :traj_length] = states.squeeze()
        self.states_buffer[next_ind, traj_length:] = 0
        self.actions_buffer[next_ind, :traj_length - 1] = actions
        self.actions_buffer[next_ind, traj_length:] = 0
        self.rewards_buffer[next_ind, :traj_length - 1] = rewards
        self.rewards_buffer[next_ind, traj_length:] = 0
        self.lens_buffer[next_ind] = traj_length
    def sample(self, batch_size):
        self.samples_since_last_training = 0
        if self.buffer_is_full: 
            indices = np.random.randint(0, self.size, batch_size)
        else: 
            indices = np.random.randint(0, self.next_spot_to_add, batch_size)
        return (self.states_buffer[indices, :, :], self.actions_buffer[indices, :],
                self.rewards_buffer[indices, :], self.lens_buffer[indices, :])
