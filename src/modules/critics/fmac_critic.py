from numpy.core.numeric import True_
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layer.self_atten import SelfAttention


class FMACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FMACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "q"
        self.hidden_states = None
        self.critic_hidden_dim = args.critic_hidden_dim 

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape + self.n_actions, self.critic_hidden_dim)
        self.fc2 = nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim)
        self.fc3 = nn.Linear(self.critic_hidden_dim, 1)

    def forward(self, inputs, actions, hidden_state=None):
        bs = inputs.batch_size
        ts = inputs.max_seq_length

        inputs = self._build_inputs(inputs)
        inputs = th.cat([inputs, actions], dim=-1)
        x = F.relu(self.fc1(inputs), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        q1 = self.fc3(x)

        return q1, hidden_state

    def _build_inputs(self, batch):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        ts = batch.max_seq_length
        inputs = []
        inputs.append(batch["obs"])  # b1av
        # inputs.append(batch["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1))  # b1av
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device)\
                .unsqueeze(0).unsqueeze(0).expand(bs, ts, -1, -1))
        inputs = th.cat([x.reshape(bs, ts, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        # input_shape += scheme["state"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape