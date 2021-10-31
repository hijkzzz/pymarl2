import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
from ..multiagentenv import MultiAgentEnv
import gym
import torch as th


class GoogleFootballEnv(MultiAgentEnv):

    def __init__(
        self,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=0,
        render=False,
        num_agents=4,
        time_limit=200,
        time_step=0,
        map_name='academy_counterattack_hard',
        stacked=False,
        representation="simple115",
        rewards='scoring,checkpoints',
        logdir='football_dumps',
        write_video=False,
        number_of_right_players_agent_controls=0,
        seed=0,
    ):
        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.n_agents = num_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.env_name = map_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed

        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
        self.env.seed(self.seed)

        obs_space_low = self.env.observation_space.low[0]
        obs_space_high = self.env.observation_space.high[0]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n
        self.obs = None

    def step(self, _actions):
        """Returns reward, terminated, info."""
        if th.is_tensor(_actions):
            actions = _actions.cpu().numpy()
        else:
            actions = _actions
        self.time_step += 1
        obs, rewards, done, infos = self.env.step(actions.tolist())

        self.obs = obs

        if self.time_step >= self.episode_limit:
            done = True

        return sum(rewards), done, infos

    def get_obs(self):
        """Returns all agent observations in a list."""
        return self.obs.reshape(self.n_agents, -1)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.obs[agent_id].reshape(-1)

    def get_obs_size(self):
        """Returns the size of the observation."""
        obs_size = np.array(self.env.observation_space.shape[1:])
        return int(obs_size.prod())

    def get_global_state(self):
        return self.obs.flatten()

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.obs = self.env.reset()

        return self.get_obs(), self.get_global_state()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_stats(self):
        return  {}
