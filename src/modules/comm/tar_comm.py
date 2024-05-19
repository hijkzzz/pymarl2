import torch.nn as nn
import torch.nn.functional as F


class TarComm(nn.Module):
    def __init__(self, input_shape, args):
        super(TarComm, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.value = nn.Linear(input_shape + args.rnn_hidden_dim, args.comm_embed_dim)
        self.signature = nn.Linear(input_shape + args.rnn_hidden_dim, args.signature_dim)
        self.query = nn.Linear(input_shape + args.rnn_hidden_dim, args.signature_dim)

    def forward(self, inputs):
        massage = self.value(inputs)
        signature = self.signature(inputs)
        query = self.query(inputs)
        return massage, signature, query
