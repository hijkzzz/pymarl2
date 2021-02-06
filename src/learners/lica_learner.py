import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.lica import LICACritic
from components.action_selectors import multinomial_entropy
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
from utils.th_utils import get_parameters_num

class LICALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.log_stats_t_agent = -self.args.learner_log_interval - 1

        self.critic = LICACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(self.mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr)

        self.entropy_coef = args.entropy_coef

        print('Mixer Size: ')
        print(get_parameters_num(self.critic.parameters()))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.train_critic_td(batch, t_env, episode_num)
        
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        mac_out = []
        mac_out_entropy = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            # -------------------------------------------------------------------------------------#
            # NOTE: We hard-coded the forward pass arguments for experiment, we will fix this later
            # -------------------------------------------------------------------------------------#
            agent_outs = self.mac.forward(batch, t=t, test_mode=True, gumbel=True)
            agent_entropy = multinomial_entropy(agent_outs).mean(dim=-1, keepdim=True)
            agent_probs = th.nn.functional.softmax(agent_outs, dim=-1)
            mac_out.append(agent_probs)
            mac_out_entropy.append(agent_entropy)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_out_entropy = th.stack(mac_out_entropy, dim=1)

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        mix_loss = self.critic(mac_out, batch["state"][:, :-1])

        mask = mask.expand_as(mix_loss)
        entropy_mask = copy.deepcopy(mask)

        mix_loss = (mix_loss * mask).sum() / mask.sum()
        entropy_loss = (mac_out_entropy * entropy_mask).sum() / entropy_mask.sum()
        entropy_ratio = self.entropy_coef / entropy_loss.item()

        mix_loss = - mix_loss - entropy_ratio * entropy_loss

        # Optimise agents
        self.agent_optimiser.zero_grad()
        mix_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if t_env - self.log_stats_t_agent >= self.args.learner_log_interval:
            self.logger.log_stat("mix_loss", mix_loss.item(), t_env)
            self.logger.log_stat("entropy", entropy_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.log_stats_t_agent = t_env


    def train_critic_td(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"]
        actions = batch["actions_onehot"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        with th.no_grad():
            # Optimise critic
            target_q_vals = self.target_critic(actions, batch["state"])[:, :]

            # Calculate td-lambda targets
            targets = build_td_lambda_targets(rewards, terminated, mask, target_q_vals, self.n_agents, self.args.gamma, self.args.td_lambda)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_t_mean": [],
        }

        mask = mask[:, :-1]

        q_t = self.critic(actions[:, :-1], batch["state"][:, :-1])

        td_error = (q_t - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm)
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_t_mean"].append((q_t * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((targets * mask).sum().item() / mask_elems)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(running_log["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_t_mean", "target_mean"]:
                self.logger.log_stat(key, sum(running_log[key])/ts_logged, t_env)
            self.log_stats_t = t_env

        if (self.critic_training_steps - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = self.critic_training_steps

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))