import copy
from torch.distributions import Categorical
from torch.optim.rmsprop import RMSprop
from components.episode_buffer import EpisodeBatch
from modules.critics.offpg import OffPGCritic
import torch as th
from utils.rl_utils import build_td_lambda_targets, build_target_q
from torch.optim import Adam
from modules.mixers.qmix import QMixer
from utils.th_utils import get_parameters_num


class OffPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = OffPGCritic(scheme, args)
        self.mixer = QMixer(args)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.mixer_params = list(self.mixer.parameters())
        self.params = self.agent_params + self.critic_params
        self.c_params = self.critic_params + self.mixer_params

        self.agent_optimiser =  RMSprop(params=self.agent_params, lr=args.lr)
        self.critic_optimiser =  RMSprop(params=self.critic_params, lr=args.lr)
        self.mixer_optimiser =  RMSprop(params=self.mixer_params, lr=args.lr)

        print('Mixer Size: ')
        print(get_parameters_num(list(self.c_params)))

    def train(self, batch: EpisodeBatch, t_env: int, log):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        states = batch["state"][:, :-1]

        #build q
        inputs = self.critic._build_inputs(batch, bs, max_t)
        q_vals = self.critic.forward(inputs).detach()[:, :-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3)
        pi = mac_out.view(-1, self.n_actions)
        baseline = th.sum(mac_out * q_vals, dim=-1).view(-1).detach()

        # Calculate policy grad with mask
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)
        coe = self.mixer.k(states).view(-1)

        advantages = (q_taken.view(-1) - baseline)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        coma_loss = - ((coe * advantages.detach() * log_pi_taken) * mask).sum() / mask.sum()
        
        # dist_entropy = Categorical(pi).entropy().view(-1)
        # dist_entropy[mask == 0] = 0 # fill nan
        # entropy_loss = (dist_entropy * mask).sum() / mask.sum()
 
        # loss = coma_loss - self.args.ent_coef * entropy_loss / entropy_loss.item()
        loss = coma_loss

        # Optimise agents
        self.agent_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        #compute parameters sum for debugging
        p_sum = 0.
        for p in self.agent_params:
            p_sum += p.data.abs().sum().item() / 100.0


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(log["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean", "q_max_mean", "q_min_mean", "q_max_var", "q_min_var"]:
                self.logger.log_stat(key, sum(log[key])/ts_logged, t_env)
            self.logger.log_stat("q_max_first", log["q_max_first"], t_env)
            self.logger.log_stat("q_min_first", log["q_min_first"], t_env)
            #self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            # self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def train_critic(self, on_batch, best_batch=None, log=None):
        bs = on_batch.batch_size
        max_t = on_batch.max_seq_length
        rewards = on_batch["reward"][:, :-1]
        actions = on_batch["actions"][:, :]
        terminated = on_batch["terminated"][:, :-1].float()
        mask = on_batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = on_batch["avail_actions"][:]
        states = on_batch["state"]

        #build_target_q
        target_inputs = self.target_critic._build_inputs(on_batch, bs, max_t)
        target_q_vals = self.target_critic.forward(target_inputs).detach()
        targets_taken = self.target_mixer(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), states)
        target_q = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda).detach()

        inputs = self.critic._build_inputs(on_batch, bs, max_t)


        if best_batch is not None:
            best_target_q, best_inputs, best_mask, best_actions, best_mac_out= self.train_critic_best(best_batch)
            log["best_reward"] = th.mean(best_batch["reward"][:, :-1].squeeze(2).sum(-1), dim=0)
            target_q = th.cat((target_q, best_target_q), dim=0)
            inputs = th.cat((inputs, best_inputs), dim=0)
            mask = th.cat((mask, best_mask), dim=0)
            actions = th.cat((actions, best_actions), dim=0)
            states = th.cat((states, best_batch["state"]), dim=0)

        #train critic
        for t in range(max_t - 1):
            mask_t = mask[:, t:t+1]
            if mask_t.sum() < 0.5:
                continue
            q_vals = self.critic.forward(inputs[:, t:t+1])
            q_ori = q_vals
            q_vals = th.gather(q_vals, 3, index=actions[:, t:t+1]).squeeze(3)
            q_vals = self.mixer.forward(q_vals, states[:, t:t+1])
            target_q_t = target_q[:, t:t+1].detach()
            q_err = (q_vals - target_q_t) * mask_t
            critic_loss = (q_err ** 2).sum() / mask_t.sum()

            self.critic_optimiser.zero_grad()
            self.mixer_optimiser.zero_grad()
            critic_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.c_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.mixer_optimiser.step()
            self.critic_training_steps += 1

            log["critic_loss"].append(critic_loss.item())
            log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            log["td_error_abs"].append((q_err.abs().sum().item() / mask_elems))
            log["target_mean"].append((target_q_t * mask_t).sum().item() / mask_elems)
            log["q_taken_mean"].append((q_vals * mask_t).sum().item() / mask_elems)
            log["q_max_mean"].append((th.mean(q_ori.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)
            log["q_min_mean"].append((th.mean(q_ori.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)
            log["q_max_var"].append((th.var(q_ori.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)
            log["q_min_var"].append((th.var(q_ori.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems)

            if (t == 0):
                log["q_max_first"] = (th.mean(q_ori.max(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems
                log["q_min_first"] = (th.mean(q_ori.min(dim=3)[0], dim=2, keepdim=True) * mask_t).sum().item() / mask_elems

        #update target network
        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps



    def train_critic_best(self, batch):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:]
        states = batch["state"]

        with th.no_grad():
            # pr for all actions of the episode
            mac_out = []
            self.mac.init_hidden(bs)
            for i in range(max_t):
                agent_outs = self.mac.forward(batch, t=i)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1).detach()
            # Mask out unavailable actions, renormalise (as in action selection)
            mac_out[avail_actions == 0] = 0
            mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
            mac_out[avail_actions == 0] = 0
            critic_mac = th.gather(mac_out, 3, actions).squeeze(3).prod(dim=2, keepdim=True)

            #target_q take
            target_inputs = self.target_critic._build_inputs(batch, bs, max_t)
            target_q_vals = self.target_critic.forward(target_inputs).detach()
            targets_taken = self.target_mixer(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), states)

            #expected q
            exp_q = self.build_exp_q(target_q_vals, mac_out, states).detach()
            # td-error
            targets_taken[:, -1] = targets_taken[:, -1] * (1 - th.sum(terminated, dim=1))
            exp_q[:, -1] = exp_q[:, -1] * (1 - th.sum(terminated, dim=1))
            targets_taken[:, :-1] = targets_taken[:, :-1] * mask
            exp_q[:, :-1] = exp_q[:, :-1] * mask
            td_q = (rewards + self.args.gamma * exp_q[:, 1:] - targets_taken[:, :-1]) * mask

            #compute target
            target_q =  build_target_q(td_q, targets_taken[:, :-1], critic_mac, mask, self.args.gamma, self.args.tb_lambda, self.args.step).detach()

            inputs = self.critic._build_inputs(batch, bs, max_t)

        return target_q, inputs, mask, actions, mac_out


    def build_exp_q(self, target_q_vals, mac_out, states):
        target_exp_q_vals = th.sum(target_q_vals * mac_out, dim=3)
        target_exp_q_vals = self.target_mixer.forward(target_exp_q_vals, states)
        return target_exp_q_vals

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.mixer.cuda()
        self.target_critic.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.mixer_optimiser.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
       # self.target_critic.load_state_dict(self.critic.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimiser.load_state_dict(th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))