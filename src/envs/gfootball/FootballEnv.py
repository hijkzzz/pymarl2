from ..multiagentenv import MultiAgentEnv
from operator import attrgetter
from copy import deepcopy
from absl import flags
import numpy as np
import pygame
import sys
import os
import math
import time
import numpy as np

import gfootball.env as football_env
from gfootball.env import config, wrappers


'''
Google Football
'''
class GoogleFootballEnv(MultiAgentEnv):
    def __init__(self, num_agents, map_name, episode_limit, seed):
        self.env = football_env.create_environment(
            # env_name='test_example_multiagent',
            env_name=map_name,  # env_name='3_vs_GK',
            representation='extracted',
            stacked=False,
            logdir='/tmp/pymarl2_fg_est' + str(seed),
            write_goal_dumps=True,
            write_full_episode_dumps=True,
            render=False,
            dump_frequency=0,
            number_of_left_players_agent_controls=num_agents,
            channel_dimensions=(42, 42))  # the preferred size for many professional teams' stadiums is 105 by 68 metres

        self.n_agents = num_agents
        self.episode_limit = episode_limit  # copied from sc2
        self.obs = None

    def step(self, actions):
        """ Returns reward, terminated, info """
        observation, reward, done, info = self.env.step(actions)
        self.obs = observation
        return np.sum(reward), done, {"score_reward": info["score_reward"]}

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        obs_agent = self.obs[agent_id].flatten()
        return obs_agent

    def get_obs_size(self):
        """ Returns the shape of the observation """
        # if obs_space is (2, 10, 10, 4) it returns (10, 10, 4)
        obs_size = np.array(self.env.observation_space.shape[1:])
        return int(obs_size.prod())

    def get_state(self):
        return self.obs.flatten()

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = np.array(self.env.observation_space.shape)
        return int(state_size.prod())

    def get_avail_actions(self):
        """Gives a representation of which actions are available to each agent.
        Returns nested list of shape: n_agents * n_actions_per_agent.
        Each element in boolean. If 1 it means that action is available to agent."""
        # assumed that all actions are available.

        total_actions = self.get_total_actions()

        avail_actions = [[1]*total_actions for i in range(0, self.n_agents)]
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id.
        Returns a list of shape: n_actions of agent.
        Each element in boolean. If 1 it means that action is available to agent."""
        # assumed that all actions are available.
        return [1]*self.get_total_actions()

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take.
        Should be integer of number of actions of an agent. Assumed that all agents have same number of actions."""
        return self.env.action_space.nvec[0]

    def get_stats(self):
        #raise NotImplementedError
        return {}

    def reset(self):
        """ Returns initial observations and states"""
        self.obs = self.env.reset()  #.reshape(self.n_agents)
        # should be return self.get_obs(), self.get_state()
        return self.get_obs(), self.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info