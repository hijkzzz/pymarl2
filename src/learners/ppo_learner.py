import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.centralv import CentralVCritic
from components.action_selectors import categorical_entropy
from utils.rl_utils import build_gae_targets
import torch as th
from torch.optim import Adam


class PPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = CentralVCritic(scheme, args)
        self.params = list(mac.parameters()) + list(self.critic.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.last_lr = args.lr
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        
        old_probs = batch["probs"][:, :-1]
        old_probs[avail_actions == 0] = 1e-10
        old_logprob = th.log(th.gather(old_probs, dim=3, index=actions)).detach()
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        
        # targets and advantages
        values = self.critic(batch)
        advantages, targets = build_gae_targets(
            rewards, mask, values, self.args.gamma, self.args.gae_lambda)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        
        for _ in range(self.args.mini_epochs):
            # Critic
            values = self.critic(batch)
            # 0-out the targets that came from padded data
            td_error = (values[:, :-1] - targets.detach())
            masked_td_error = td_error * mask
            critic_loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()

            # Actor
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.mac.forward(batch, t=t)
                pi.append(agent_outs)
            pi = th.stack(pi, dim=1)  # Concat over time

            pi[avail_actions == 0] = 1e-10
            pi_taken = th.gather(pi, dim=3, index=actions)
            log_pi_taken = th.log(pi_taken)
            
            ratios = th.exp(log_pi_taken - old_logprob)
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            actor_loss = -(th.min(surr1, surr2) * mask_agent).sum() / mask_agent.sum()
            
            # entropy
            entropy_loss = categorical_entropy(pi).mean(-1, keepdim=True) # mean over agents
            entropy_loss[mask == 0] = 0 # fill nan
            entropy_loss = (entropy_loss* mask).sum() / mask.sum()
            loss = actor_loss + self.args.critic_coef * critic_loss - self.args.entropy * entropy_loss / entropy_loss.item()

            # Optimise agents
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()
            

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("advantage_mean", (advantages * mask_agent).sum().item() / mask_agent.sum().item(), t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("lr", self.last_lr, t_env)
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / mask_elems, t_env)
            self.log_stats_t = t_env


    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/agent_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
