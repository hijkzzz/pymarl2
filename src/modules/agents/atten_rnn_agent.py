import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from modules.layer.self_atten import SelfAttention

class ATTRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ATTRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.att = SelfAttention(input_shape, args.att_heads, args.att_embed_dim)
        self.fc2 = nn.Linear(args.att_heads *  args.att_embed_dim, args.rnn_hidden_dim)

        self.fc3 = nn.Sequential(nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(args.rnn_hidden_dim, args.n_actions))

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # INPUT
        b, a, e = inputs.size()

        # RNN
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        # ATT
        att = self.att(inputs.view(b, a, -1))
        att = F.relu(self.fc2(att), inplace=True).view(-1, self.args.rnn_hidden_dim)

        # Q
        q = th.cat((h, att), dim=-1)
        q = self.fc3(q)

        return q.view(b, a, -1), h.view(b, a, -1)