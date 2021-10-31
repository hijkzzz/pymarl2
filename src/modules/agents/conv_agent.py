import torch.nn as nn
import torch.nn.functional as F


class ConvAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ConvAgent, self).__init__()
        self.args = args
        self.hidden_dim = self.args.rnn_hidden_dim

        self.conv1 = nn.Conv1d(input_shape, self.hidden_dim, 2)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, 3)
        self.linear_hidden_dim = (self.args.frames - 3) * self.hidden_dim

        self.fc1 = nn.Linear(self.linear_hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, args.n_actions)

    def init_hidden(self):
        return None

    def forward(self, inputs, hidden_state=None):
        b, t, a, c = inputs.size()
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, c, t)
        
        x = F.relu(self.conv1(inputs), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = x.view(b, a, self.linear_hidden_dim)
        x = F.relu(self.fc1(x), inplace=True)
        q = self.fc2(x)
        
        return q.view(b, a, -1), None