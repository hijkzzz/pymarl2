import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np
from collections import deque

# Source Code: https://github.com/AnujMahajanOxf/MAVEN
class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.state_shape, 128)
        self.fc2 = nn.Linear(128, args.noise_dim)

    def forward(self, x):
        x = x.reshape(-1, self.args.state_shape)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class EZAgent():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.noise_dim = args.noise_dim
        self.state_shape = args.state_shape
        self.lr = args.lr

        self.policy = Policy(args)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.entropy_scaling = args.entropy_scaling
        self.uniform_distrib = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([1/self.args.noise_dim for _ in range(self.args.noise_dim)]).repeat(self.args.batch_size_run, 1))

        self.buffer = deque(maxlen=self.args.bandit_buffer)
        self.epsilon_floor = args.bandit_epsilon

    def sample(self, state, test_mode):
        if test_mode:
            return self.uniform_distrib.sample()
        else:
            probs = self.policy(state)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
            action = m.sample().cpu()
            return action

    def update_returns(self, states, actions, returns, test_mode, t):
        if test_mode:
            return

        for s,a,r in zip(states, actions, returns):
            self.buffer.append((s,a,torch.tensor(r, dtype=torch.float)))

        for _ in range(self.args.bandit_iters):
            idxs = np.random.randint(0, len(self.buffer), size=self.args.bandit_batch)
            batch_elems = [self.buffer[i] for i in idxs]
            states_ = torch.stack([x[0] for x in batch_elems]).to(states.device)
            actions_ = torch.stack([x[1] for x in batch_elems]).to(states.device)
            returns_ = torch.stack([x[2] for x in batch_elems]).to(states.device)

            probs = self.policy(states_)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
            log_probs = m.log_prob(actions_.to(probs.device))
            self.optimizer.zero_grad()
            policy_loss = -torch.dot(log_probs, torch.tensor(returns_, device=log_probs.device).float()) + self.entropy_scaling * log_probs.sum()
            policy_loss.backward()
            self.optimizer.step()

        mean_entropy = m.entropy().mean()
        self.logger.log_stat("bandit_entropy", mean_entropy.item(), t)

    def cuda(self):
        self.policy.cuda()

    def save_model(self, path):
        torch.save(self.policy.state_dict(), "{}/ez_bandit_policy.th".format(path))


class Discrim(nn.Module):

    def __init__(self, input_size, output_size, args):
        super().__init__()
        self.args = args
        layers = [nn.Linear(input_size, self.args.discrim_size), nn.ReLU()]
        for _ in range(self.args.discrim_layers - 1):
            layers.append(nn.Linear(self.args.discrim_size, self.args.discrim_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.args.discrim_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RNNAggregator(nn.Module):

    def __init__(self, input_size, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        output_size = args.rnn_agg_size
        self.rnn = nn.GRUCell(input_size, output_size)

    def forward(self, x, h):
        return self.rnn(x, h)