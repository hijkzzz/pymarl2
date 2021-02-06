import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Sequential(nn.Linear(input_shape, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256, 1)
                                 )

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        q = self.fc1(inputs)
        return q

    def _build_inputs(self, batch, t=None):
        ts = slice(None) if t is None else slice(t, t+1)
        return batch["state"][:, ts]

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        return input_shape