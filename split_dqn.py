"""
Deep Q-Network
==============
"""
import numpy as np
import pickle
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

from clt_core.core import Agent
from clt_core.util.memory import ReplayBuffer

from dqn import QNetwork


def split_replay_buffer(buffer, nr_buffers, nr_rotations):
    """ Splits a buffer with mixed transitions (from different primitives) to
    one buffer per primitive.
    """
    result = []
    for _ in range(nr_buffers):
        result.append(ReplayBuffer(1e6))
    for i in range(buffer.size()):
        result[int(np.floor(buffer(i).action / nr_rotations))].store(buffer(i))
    return result


class SplitDQN(Agent):
    def __init__(self, state_dim, action_dim, params):
        super().__init__('split_dqn', params)
        self.action_dim = action_dim
        self.state_dim = state_dim

        # The number of networks is the number of high level actions (e.g. push
        # target, push obstacles, grasp). One network per high level action.
        self.nr_networks = int(len(self.params['hidden_units']))

        # Nr of substates is the number of low level actions, which are
        # represented as different states (e.g. rotations of visual features).
        self.nr_rotations = int(self.action_dim / self.nr_networks)

        self.device = self.params['device']

        torch.manual_seed(0)

        # Create a list of networks and their targets
        self.network, self.target_network = nn.ModuleList(), nn.ModuleList()
        for hidden in self.params['hidden_units']:
            self.network.append(QNetwork(self.state_dim, 1, hidden).to(self.device))
            self.target_network.append(QNetwork(self.state_dim, 1, hidden).to(self.device))

        # Set the optimizer and loss for each primitive
        self.optimizer = []
        self.loss = []
        self.replay_buffer = []
        self.save_buffer = []
        for i in range(self.nr_networks):
            self.optimizer.append(optim.Adam(self.network[i].parameters(),
                                             lr=self.params['learning_rate'][i]))

            if self.params['loss'][i] == 'mse':
                self.loss.append(nn.MSELoss())
            elif self.params['loss'][i] == 'huber':
                self.loss.append(nn.SmoothL1Loss())
            else:
                raise ValueError('SplitDQN: Loss should be mse or huber')
            self.info['qnet_' + str(i) + '_loss'] = 0

            self.replay_buffer.append(ReplayBuffer(self.params['replay_buffer_size']))
            self.save_buffer.append(False)

        self.learn_step_counter = 0
        self.rng = np.random.RandomState()
        self.epsilon = self.params['epsilon_start']
        self.info['epsilon'] = 0.0

    def predict(self, state):
        action_value = []
        for i in range(self.nr_networks):
            for j in range(self.nr_rotations):
                s = torch.FloatTensor(state[j]).to(self.device)
                action_value.append(self.network[i](s).cpu().detach().numpy())
        return np.argmax(action_value)

    def explore(self, state):
        self.epsilon = self.params['epsilon_end'] + (self.params['epsilon_start'] - self.params['epsilon_end']) * \
                       math.exp(-1 * self.learn_step_counter / self.params['epsilon_decay'])
        if self.rng.uniform(0, 1) >= self.epsilon:
            return self.predict(state)
        else:
            return self.rng.randint(0, self.action_dim)

    def learn(self, transition):
        i = int(np.floor(transition.action / self.nr_rotations))
        self.replay_buffer[i].store(transition)
        self.update_net(i)

        self.info['epsilon'] = self.epsilon  # save for plotting

    def update_net(self, i):
        self.info['qnet_' + str(i) + '_loss'] = 0

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit.
        if self.replay_buffer[i].size() < self.params['init_replay_buffer_size']:
            return

        if not self.save_buffer[i]:
            self.replay_buffer[i].save(os.path.join(self.params['log_dir'], 'replay_buffer_' + str(i)))
            self.save_buffer[i] = True

        # Update target net's params
        new_target_params = {}
        for key in self.target_network[i].state_dict():
            new_target_params[key] = self.params['tau'] * self.target_network[i].state_dict()[key] + \
                                     (1 - self.params['tau']) * self.network[i].state_dict()[key]
        self.target_network[i].load_state_dict(new_target_params)

        # Sample from replay buffer
        batch = self.replay_buffer[i].sample_batch(self.params['batch_size'][i])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))
        batch.action = np.array(batch.action.reshape((batch.action.shape[0], 1)))

        next_state = torch.FloatTensor(batch.next_state).to(self.params['device'])
        terminal = torch.FloatTensor(batch.terminal).to(self.params['device'])
        reward = torch.FloatTensor(batch.reward).to(self.params['device'])

        # Calculate maxQ(s_next, a_next) with max over next actions
        q_next = torch.FloatTensor().to(self.device)
        for net in range(self.nr_networks):
            for k in range(self.nr_rotations):
                q_next_i = self.target_network[net](next_state[:, k])
                q_next = torch.cat((q_next, q_next_i), dim=1)
        q_next = q_next.max(1)[0].view(self.params['batch_size'][i], 1)

        q_target = reward + (1 - terminal) * self.params['discount'] * q_next

        # Calculate current q
        st = np.zeros((self.params['batch_size'][i], self.state_dim))
        batch.action = np.subtract(batch.action, i * self.nr_rotations)
        for m in range(self.params['batch_size'][i]):
            st[m] = batch.state[m, batch.action[m]]
        s = torch.FloatTensor(st).to(self.device)
        q = self.network[i](s)

        loss = self.loss[i](q, q_target)
        self.optimizer[i].zero_grad()
        loss.backward()
        self.optimizer[i].step()
        self.info['qnet_' + str(i) + '_loss'] = loss.detach().cpu().numpy().copy()

        self.learn_step_counter += 1

    def q_value(self, state, action):
        net_index = int(np.floor(action / self.nr_rotations))
        rotation_index = int(action - np.floor(action / self.nr_rotations) * self.nr_rotations)
        s = torch.FloatTensor(state[rotation_index]).to(self.device)
        return self.network[net_index](s).cpu().detach().numpy()

    def seed(self, seed):
        for i in range(len(self.replay_buffer)):
            self.replay_buffer[i].seed(seed)
        self.rng.seed(seed)

    def save(self, save_dir, name):
        # Create directory
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir)

        for i in range(self.nr_networks):
            # Save networks and log data
            torch.save({'network': self.network[i].state_dict(),
                        'target_network': self.target_network[i].state_dict()},
                       os.path.join(log_dir, 'model_' + str(i) + '.pt'))
        log_data = {'params': self.params.copy(),
                    'learn_step_counter': self.learn_step_counter,
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim}
        pickle.dump(log_data, open(os.path.join(log_dir, 'log_data.pkl'), 'wb'))

    @classmethod
    def load(cls, log_dir):
        log_data = pickle.load(open(os.path.join(log_dir, 'log_data.pkl'), 'rb'))
        self = cls(state_dim=log_data['state_dim'],
                   action_dim=log_data['action_dim'],
                   params=log_data['params'])

        def get_model_id(x): return int(x.split('.')[0].split('_')[-1])

        for file in os.listdir(log_dir):
            if file.endswith('.pt'):
                i = get_model_id(file)

                checkpoint = torch.load(os.path.join(log_dir, file))
                self.network[i].load_state_dict(checkpoint['network'])
                self.target_network[i].load_state_dict(checkpoint['target_network'])

        return self
