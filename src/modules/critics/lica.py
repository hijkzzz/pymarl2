import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LICACritic(nn.Module):
    def __init__(self, scheme, args):
        super(LICACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.output_type = "q"

        # Set up network layers
        self.state_dim = int(np.prod(args.state_shape))
        self.weight_dim = args.lica_mixing_embed_dim * self.n_agents * self.n_actions
        self.hid_dim = args.hypernet_embed_dim

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hid_dim , self.weight_dim))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hid_dim, args.lica_mixing_embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, args.lica_mixing_embed_dim)

        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                               nn.ReLU(),
                               nn.Linear(self.hid_dim, 1))

    def forward(self, act, states):
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim)
        action_probs = act.reshape(-1, 1, self.n_agents * self.n_actions)

        # first layer
        w1 = self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents * self.n_actions, self.args.lica_mixing_embed_dim)
        b1 = b1.view(-1, 1, self.args.lica_mixing_embed_dim)

        h = th.relu(th.bmm(action_probs, w1) + b1)

        # second layer
        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.args.lica_mixing_embed_dim, 1)
        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        q = th.bmm(h, w_final )+ b2
        q = q.view(bs, -1, 1)

        return q