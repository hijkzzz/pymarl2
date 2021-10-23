import torch.nn as nn
import torch.nn.functional as F
from utils.noisy_liner import NoisyLinear
from torch.nn import LayerNorm

class NoisyRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NoisyRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = NoisyLinear(args.rnn_hidden_dim, args.n_actions, True, args.device)


        if getattr(args, "use_feature_norm", False):
            self.feature_norm = LayerNorm(input_shape)
        

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        
        inputs = inputs.view(-1, e)
        if getattr(self.args, "use_feature_norm", False):
            inputs = self.feature_norm(inputs)
        
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q.view(b, a, -1), h.view(b, a, -1)