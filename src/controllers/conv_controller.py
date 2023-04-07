from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class ConvMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(ConvMAC, self).__init__(scheme, groups, args)
        self.buffer = []
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode).squeeze(-1)
        return self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)

    def forward(self, ep_batch, t, test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t]
        with th.no_grad():
            agent_inputs = self._build_inputs(ep_batch, t)
            if len(self.buffer) < self.args.frames:
                self.buffer = [th.zeros_like(agent_inputs) for i in range(self.args.frames)]
            # stack
            self.buffer = self.buffer[1:] + [agent_inputs]  
            batch_agent_inputs = th.stack(self.buffer, dim=1) # b, t, a, c
        
        agent_outs, self.hidden_states = self.agent(batch_agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e5

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        super(ConvMAC, self).init_hidden(batch_size)
        self.buffer = []