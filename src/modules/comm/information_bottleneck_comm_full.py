import torch as th
import torch.nn as nn
import torch.nn.functional as F


class IBFComm(nn.Module):
	def __init__(self, input_shape, args):
		super(IBFComm, self).__init__()
		self.args = args
		self.n_agents = args.n_agents

		self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
		self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
		self.fc3 = nn.Linear(args.rnn_hidden_dim, args.comm_embed_dim * self.n_agents)

		self.inference_model = nn.Sequential(
			nn.Linear(input_shape + args.comm_embed_dim * self.n_agents, 4 * args.comm_embed_dim * self.n_agents),
			nn.ReLU(True),
			nn.Linear(4 * args.comm_embed_dim * self.n_agents, 4 * args.comm_embed_dim * self.n_agents),
			nn.ReLU(True),
			nn.Linear(4 * args.comm_embed_dim * self.n_agents, args.n_actions)
		)

	def forward(self, inputs):
		x = F.relu(self.fc1(inputs))
		x = F.relu(self.fc2(x))
		gaussian_params = self.fc3(x)

		mu = gaussian_params
		#sigma = F.softplus(gaussian_params[:, self.args.comm_embed_dim * self.n_agents:])
		sigma = th.ones(mu.shape).cuda()

		return mu, sigma
