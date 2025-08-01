import math
import copy
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drl_baseline.common_cfg import DEFAULT_AGENT_CONFIG, DEFAULT_MEMORY_CONFIG, DEFAULT_TRAIN_CONFIG
from drl_baseline.memory import ReplayMemory, PrioritizedReplayMemory


class DRLAgent:

    def __init__(self,
                 agent_config=None,
                 memory_config=None,
                 train_config=None):

        if agent_config is None:
            self.agent_config = DEFAULT_AGENT_CONFIG
        else:
            self.agent_config = agent_config
        if memory_config is None:
            self.memory_config = DEFAULT_MEMORY_CONFIG
        else:
            self.memory_config = memory_config
        if train_config is None:
            self.train_config = DEFAULT_TRAIN_CONFIG
        else:
            self.train_config = train_config

        self.device = train_config['device']

        assert self.agent_config['env']['state_dim'] is not None, 'state dimension is not set!'
        assert self.agent_config['env']['action_dim'] is not None, 'action dimension is not set!'

        self.gamma = self.agent_config['base']['gamma']  # discounted factor
        # tau is used for polyak update agent networks (DQN, DDPG, TD3), =1 means totally eval net -> target net
        self.tau = self.agent_config['base']['tau']

        memory_class = PrioritizedReplayMemory if self.agent_config['memory']['prior_enable'] else ReplayMemory

        self.replay_memory = memory_class(max_size=self.agent_config['memory']['max_size'],
                                          batch_size=self.agent_config['memory']['batch_size'],  # training batch size
                                          memory_info=self.memory_config,
                                          use_latest=self.agent_config['memory']['use_latest'],
                                          device=self.device)  # on-policy or off-policy experience buffer
        self.replay_memory.reset_memory()

    @abstractmethod
    def net_construct(self, **kwargs):
        raise NotImplementedError

    def store_data(self, **kwargs):
        """

        Args:
            **kwargs:

        Returns:

        """
        self.replay_memory.store_data(**kwargs)

    @abstractmethod
    def choose_action(self, state, deterministic=False):
        """
        Interact with env, and choose target action.

        Args:
            state: agent state
            deterministic: if True,

        Returns:
            action: discrete type action (0,1,2,3,4....) or continuous action [-1,1]*n

        """
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        """
        Learn to update the agent, should contain buffer sample and learning procedure.

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def soft_update_net_paras(source_net, target_net, polyak_tau):
        for param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.copy_(polyak_tau * param.data + (1 - polyak_tau) * target_param.data)

    @abstractmethod
    def save(self, path):
        """
        Save a checkpoint the agent.

        Args:
            path: file path

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        """
        Load a checkpoint the agent.

        Args:
            path: file path

        Returns:

        """
        raise NotImplementedError


class Configer(ABC):

    def __init__(self):
        self.agent_config = copy.deepcopy(DEFAULT_AGENT_CONFIG)
        self.memory_config = copy.deepcopy(DEFAULT_MEMORY_CONFIG)
        self.train_config = copy.deepcopy(DEFAULT_TRAIN_CONFIG)

    @abstractmethod
    def specify_config(self, env) -> Tuple[Dict, ...]:
        """
        Given env, setup specific configurations for a target DRL agent

        Args:
            env:

        Returns:
            agent_config, memory_config, train_config
        """
        raise NotImplementedError
