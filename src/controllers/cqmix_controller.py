from gym import spaces
import torch as th
import torch.distributions as tdist
import numpy as np

from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class CQMixMAC(BasicMAC):
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if self.args.agent in ["naf", "mlp"]:
            chosen_actions = self.forward(ep_batch[bs], t_ep, test_mode=test_mode, select_actions=True)
            chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()
        elif self.args.agent in ["cem"]:
            chosen_actions = self.cem_sampling(ep_batch, t_ep, bs)
        else:
            raise Exception("No known agent type selected! ({})".format(self.args.agent))

        # Now do appropriate exploration
        exploration_mode = getattr(self.args, "exploration_mode", "gaussian")
        if not test_mode:
            if exploration_mode == "ornstein_uhlenbeck":
                x = getattr(self, "ou_noise_state", chosen_actions.clone().zero_())
                mu = 0
                theta = getattr(self.args, "ou_theta", 0.15)
                sigma = getattr(self.args, "ou_sigma", 0.2)
                noise_scale = getattr(self.args, "ou_noise_scale", 0.3) if t_env < self.args.env_args["episode_limit"]*self.args.ou_stop_episode else 0.0

                dx = theta * (mu - x) + sigma * x.clone().normal_()
                self.ou_noise_state = x + dx
                ou_noise = self.ou_noise_state * noise_scale
                chosen_actions = chosen_actions + ou_noise
            elif exploration_mode == "gaussian":
                start_steps = getattr(self.args, "start_steps", 0)
                act_noise = getattr(self.args, "act_noise", 0.1)
                if t_env >= start_steps:
                    x = chosen_actions.clone().zero_()
                    chosen_actions += act_noise * x.clone().normal_()
                else:
                    if self.args.env_args["scenario"] in ["Humanoid-v2", "HumanoidStandup-v2"]:
                        chosen_actions = th.from_numpy(np.array([self.args.action_spaces[0].sample() for i in range(self.n_agents)])).unsqueeze(0).float().to(device=ep_batch.device)
                    else:
                        chosen_actions = th.from_numpy(np.array([self.args.action_spaces[i].sample() for i in range(self.n_agents)])).unsqueeze(0).float().to(device=ep_batch.device)

        # now clamp actions to permissible action range (necessary after exploration)
        if all([isinstance(act_space, spaces.Box) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(np.asscalar(self.args.action_spaces[_aid].low[_actid]),
                                                           np.asscalar(self.args.action_spaces[_aid].high[_actid]))
        elif all([isinstance(act_space, spaces.Tuple) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].spaces[0].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(self.args.action_spaces[_aid].spaces[0].low[_actid],
                                                           self.args.action_spaces[_aid].spaces[0].high[_actid])
                for _actid in range(self.args.action_spaces[_aid].spaces[1].shape[0]):
                    tmp_idx = _actid + self.args.action_spaces[_aid].spaces[0].shape[0]
                    chosen_actions[:, _aid, tmp_idx].clamp_(self.args.action_spaces[_aid].spaces[1].low[_actid],
                                                            self.args.action_spaces[_aid].spaces[1].high[_actid])
        return chosen_actions

    def get_weight_decay_weights(self):
        return self.agent.get_weight_decay_weights()

    def forward(self, ep_batch, t, actions=None, select_actions=False, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        ret = self.agent(agent_inputs, actions=actions)
        if select_actions:
            return ret
        agent_outs = ret["Q"]

        if self.agent_output_type == "pi_logits":
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon/agent_outs.size(-1))
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), actions

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def cem_sampling(self, ep_batch, t, bs):
        # Number of samples from the param distribution
        N = 64
        # Number of best samples we will consider
        Ne = 6

        ftype = th.FloatTensor if not next(self.agent.parameters()).is_cuda else th.cuda.FloatTensor
        mu = ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_()
        std = ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_() + 1.0
        its = 0
        maxits = 2
        agent_inputs = self._build_inputs(ep_batch[bs], t)

        while its < maxits:
            dist = tdist.Normal(mu.view(-1, self.args.n_actions), std.view(-1, self.args.n_actions))
            actions = dist.sample((N,)).detach()
            actions_prime = th.tanh(actions)
            ret = self.agent(agent_inputs.unsqueeze(0).expand(N, *agent_inputs.shape).contiguous().view(-1, agent_inputs.shape[-1]),
                             actions=actions_prime.view(-1, actions_prime.shape[-1]))
            out = ret["Q"].view(N, -1, 1)
            topk, topk_idxs = th.topk(out, Ne, dim=0)
            mu = th.mean(actions.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
            std = th.std(actions.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
            its += 1
        topk, topk_idxs = th.topk(out, 1, dim=0)
        action_prime = th.mean(actions_prime.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
        chosen_actions = action_prime.clone().view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()
        return chosen_actions