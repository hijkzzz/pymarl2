import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
import torch.distributions as D


class CateQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.s_mu = th.zeros(1)
        self.s_sigma = th.ones(1)

    def get_comm_beta(self, t_env):
        comm_beta = self.args.comm_beta
        if self.args.is_comm_beta_decay and t_env > self.args.comm_beta_start_decay:
            comm_beta += 1. * (self.args.comm_beta_target - self.args.comm_beta) / \
                         (self.args.comm_beta_end_decay - self.args.comm_beta_start_decay) * \
                         (t_env - self.args.comm_beta_start_decay)
        return comm_beta

    def get_comm_entropy_beta(self, t_env):
        comm_entropy_beta = self.args.comm_entropy_beta
        if self.args.is_comm_entropy_beta_decay and t_env > self.args.comm_entropy_beta_start_decay:
            comm_entropy_beta += 1. * (self.args.comm_entropy_beta_target - self.args.comm_entropy_beta) / \
                                 (self.args.comm_entropy_beta_end_decay - self.args.comm_entropy_beta_start_decay) * \
                                 (t_env - self.args.comm_entropy_beta_start_decay)
        return comm_entropy_beta

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        # shape = (bs, self.n_agents, -1)
        mac_out = []
        mu_out = []
        sigma_out = []
        logits_out = []
        m_sample_out = []
        g_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if self.args.comm and self.args.use_IB:
                agent_outs, (mu, sigma), logits, m_sample = self.mac.forward(batch, t=t)
                mu_out.append(mu)
                sigma_out.append(sigma)
                logits_out.append(logits)
                m_sample_out.append(m_sample)
            else:
                agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        if self.args.use_IB:
            mu_out = th.stack(mu_out, dim=1)[:, :-1]  # Concat over time
            sigma_out = th.stack(sigma_out, dim=1)[:, :-1]  # Concat over time
            logits_out = th.stack(logits_out, dim=1)[:, :-1]
            m_sample_out = th.stack(m_sample_out, dim=1)[:, :-1]

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # I believe that code up to here is right...

        # Q values are right, the main issue is to calculate loss for message...

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if self.args.comm and self.args.use_IB:
                target_agent_outs, (target_mu, target_sigma), target_logits, target_m_sample = \
                    self.target_mac.forward(batch, t=t)
            else:
                target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # label
        label_target_max_out = th.stack(target_mac_out[:-1], dim=1)
        label_target_max_out[avail_actions[:, :-1] == 0] = -9999999
        label_target_actions = label_target_max_out.max(dim=3, keepdim=True)[1]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            # mac_out[avail_actions == 0] = -9999999
            # cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
            # target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach().clone())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.only_downstream or not self.args.use_IB:
            expressiveness_loss = th.Tensor([0.])
            compactness_loss = th.Tensor([0.])
            entropy_loss = th.Tensor([0.])
            comm_loss = th.Tensor([0.])
            comm_beta = th.Tensor([0.])
            comm_entropy_beta = th.Tensor([0.])
        else:
            # ### Optimize message
            # Message are controlled only by expressiveness and compactness loss.
            # Compute cross entropy with target q values of the same time step
            expressiveness_loss = 0
            label_prob = th.gather(logits_out, 3, label_target_actions).squeeze(3)
            expressiveness_loss += (-th.log(label_prob + 1e-6)).sum() / mask.sum()

            # Compute KL divergence
            compactness_loss = D.kl_divergence(D.Normal(mu_out, sigma_out), D.Normal(self.s_mu, self.s_sigma)).sum() / \
                               mask.sum()

            # Entropy loss
            entropy_loss = -D.Normal(self.s_mu, self.s_sigma).log_prob(m_sample_out).sum() / mask.sum()

            # Gate loss
            gate_loss = 0

            # Total loss
            comm_beta = self.get_comm_beta(t_env)
            comm_entropy_beta = self.get_comm_entropy_beta(t_env)
            comm_loss = expressiveness_loss + comm_beta * compactness_loss + comm_entropy_beta * entropy_loss
            comm_loss *= self.args.c_beta
            loss += comm_loss
            comm_beta = th.Tensor([comm_beta])
            comm_entropy_beta = th.Tensor([comm_entropy_beta])

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Update target
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("comm_loss", comm_loss.item(), t_env)
            self.logger.log_stat("exp_loss", expressiveness_loss.item(), t_env)
            self.logger.log_stat("comp_loss", compactness_loss.item(), t_env)
            self.logger.log_stat("comm_beta", comm_beta.item(), t_env)
            self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("comm_beta", comm_beta.item(), t_env)
            self.logger.log_stat("comm_entropy_beta", comm_entropy_beta.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    # self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.s_mu = self.s_mu.cuda()
        self.s_sigma = self.s_sigma.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
