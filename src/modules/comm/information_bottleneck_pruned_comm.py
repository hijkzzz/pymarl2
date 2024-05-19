import torch.nn as nn
import torch.nn.functional as F


class IBPComm(nn.Module):
    def __init__(self, input_shape, args):
        super(IBPComm, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.comm_embed_dim * 2)

        self.inference_model = nn.Sequential(
            nn.Linear(args.comm_embed_dim, 2*args.comm_embed_dim),
            nn.ReLU(True),
            nn.Linear(2*args.comm_embed_dim, 2 * args.comm_embed_dim),
            nn.ReLU(True),
            nn.Linear(2 * args.comm_embed_dim, args.atom)
        )

        self.gate = nn.Sequential(
            nn.Linear(args.comm_embed_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, args.comm_embed_dim)
        )

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        gaussian_params = self.fc2(x)

        mu = gaussian_params[:, :self.args.comm_embed_dim]
        sigma = F.softplus(gaussian_params[:, self.args.comm_embed_dim:])

        return mu, sigma
