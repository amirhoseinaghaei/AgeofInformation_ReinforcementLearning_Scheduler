from RUDDER.nn import LSTMLayer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import MSELoss as MSELoss
from Preprocessing import to_one_hot


class RRLSTM(nn.Module):
    def __init__(self, state_input_size, n_actions, buffer, n_units, lstm_lr, l2_regularization,
                 return_scaling, lstm_batch_size=128, continuous_pred_factor=0.5):
        super(RRLSTM, self).__init__()
        self.buffer = buffer
        self.return_scaling = return_scaling
        self.lstm_batch_size = lstm_batch_size
        self.continuous_pred_factor = continuous_pred_factor
        self.n_actions = n_actions    
        # Forget gate and output gate are deactivated as used in the Atari games, see Appendix S4.2.1
        self.lstm = LSTMLayer(in_features=state_input_size + n_actions, out_features=n_units,
                              w_ci=(lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs), False),
                              w_ig=(False, lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs)),
                              # w_og=False,
                              w_og=(lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs), False), 
                              b_ci=lambda *args, **kwargs: nn.init.normal_(mean=0, *args, **kwargs),
                              b_ig=lambda *args, **kwargs: nn.init.normal_(mean=-3, *args, **kwargs),
                              # b_og=False,
                              b_og=lambda *args, **kwargs: nn.init.normal_(mean=0, *args, **kwargs),
                              a_out=lambda x: x
                              )
        self.linear = nn.Linear(n_units, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lstm_lr, weight_decay=l2_regularization)
        self.lstm_updates = 0
    def forward(self, input):
        states, actions = input
        actions = to_one_hot(actions, self.n_actions)
        actions = torch.cat((actions, torch.zeros((actions.shape[0], 1, self.n_actions))), 1)
        input = torch.cat((states, actions), 2)
        # Run the lstm
        lstm_out = self.lstm.forward(input, return_all_seq_pos=True)
        return self.linear(lstm_out[0])
    def redistribute_reward(self, states, actions):
        # Prepare LSTM inputs
        states_var = Variable(torch.FloatTensor(states)).detach()
        delta_states = torch.cat([states_var[:, 0:1, :], states_var[:, 1:, :] - states_var[:, :-1, :]], dim=1)
        actions_var = Variable(torch.FloatTensor(actions)).detach()
        # Calculate LSTM predictions
        lstm_out = self.forward([delta_states, actions_var])
        pred_g0 = torch.cat([torch.zeros_like(lstm_out[:, 0:1, :]), lstm_out], dim=1)[:, :-1, :]
        # Difference of predictions of two consecutive timesteps.
        redistributed_reward = pred_g0[:, 1:, 0] - pred_g0[:, :-1, 0]
        new_reward = redistributed_reward * self.return_scaling
        return new_reward

    # Trains the LSTM until -on average- the main loss is below 0.25.
    def train(self, episode):
        # print("HI")
        i = 0
        loss_average = 0.15
        mse_loss = MSELoss(reduction="none")
        while loss_average > 0.1:
            # # print(loss_average)
            # if loss_average < 5:
            #     print(loss_average)
            i += 1
            self.lstm_updates += 1
            self.optimizer.zero_grad()

            # Get samples from the lesson buffer and prepare them.
            states, actions, rewards, lenght = self.buffer.sample(self.lstm_batch_size)
            lenght = lenght[:, 0]
            states_var = Variable(torch.FloatTensor(states)).detach()
            actions_var = Variable(torch.FloatTensor(actions)).detach()
            rewards_var = Variable(torch.FloatTensor(rewards)).detach()

            # Scale the returns as they might have high / low values.
            returns = torch.sum(rewards_var, 1, keepdim=True) / self.return_scaling

            # Calculate differences of states
            delta_states = torch.cat([states_var[:, 0:1, :], states_var[:, 1:, :] - states_var[:, :-1, :]], dim=1)

            # Run the LSTM
            lstm_out = self.forward([delta_states, actions_var])
            predicted_G0 = lstm_out.squeeze()

            # Loss calculations
            all_timestep_loss = mse_loss(predicted_G0, returns.repeat(1, predicted_G0.size(1)))

            # Loss at any position in the sequence
            aux_loss = self.continuous_pred_factor * all_timestep_loss.mean()

            # # LSTM is mainly trained on getting the final prediction of g0 right.
            # print("**********************************************************************")
            # # print(predicted_G0)
            # # print(all_timestep_loss)
            # print(all_timestep_loss[range(self.lstm_batch_size), lenght[:] - 1])
            # print(range(self.lstm_batch_size))
            # print(lenght[:] - 1)
            main_loss = all_timestep_loss[range(self.lstm_batch_size), lenght[:] - 1].mean()

            # LSTM update and loss tracking
            lstm_loss = main_loss + aux_loss
            lstm_loss.backward()
            loss_np = lstm_loss.data.numpy()
            main_loss_np = main_loss.data.numpy()
            loss_average -= 0.01 * (loss_average - main_loss_np)
            if main_loss_np > loss_average * 2:
                loss_average = loss_np
            self.optimizer.step()
        # print("BYE")
