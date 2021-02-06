# From https://github.com/wjh720/QPLEX/, added here for convenience.
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dmaq_si_weight import DMAQ_SI_Weight


class DMAQer(nn.Module):
    def __init__(self, args):
        super(DMAQer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.embed_dim = args.mixing_embed_dim

        hypernet_embed = self.args.hypernet_embed
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents))
        self.V = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, self.n_agents))

        self.si_weight = DMAQ_SI_Weight(args)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)

        w_final = self.hyper_w_final(states)
        w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = self.V(states)
        v = v.view(-1, self.n_agents)

        if self.args.weighted_head:
            agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            if self.args.weighted_head:
                max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot
