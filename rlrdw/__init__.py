from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from drl_baseline import DRLAgent

from pyrdw import *
import pyrdw.lib.math.algebra as alg
import pyrdw.lib.math.geometry as geo
import pyrdw.lib.math.image as img

from pyrdw.core import PI, PI_1_2, PI_2
from pyrdw.core.agent.base import GeneralRdwAgent
from pyrdw.core.space.trajectory import Trajectory
from pyrdw.core.space.scene import DiscreteScene
from pyrdw.core.env.base import RdwEnv

from common import project_path, data_path, rl_model_path


class DRLRDWAgent(GeneralRdwAgent):

    def __init__(self):
        super().__init__()
        self.drl_name = 'DRL'
        self.drl_agent: DRLAgent = None
        self.drl_action = None
        self.drl_state = None
        self.drl_history = deque([], maxlen=25)  # used for action repeat drl (srl, src, etc.)
        self.drl_freeze = False
        self.max_epi_step = 0
        self.epi_step = 0

    @abstractmethod
    def action_transition(self, *args):
        """
        Transform the action from drl agent to the local agent

        Args:
            act: action output of drl_agent

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, state, deterministic=False):
        raise NotImplementedError

    # @abstractmethod
    # def step_from_action(self, act):
    #     raise NotImplementedError

    @abstractmethod
    def step_non_deterministic(self):
        raise NotImplementedError

    def learn(self):
        return self.drl_agent.learn()

    def store_step_data(self, **kwargs):
        """

        Args:
            **kwargs:

        Returns:

        """
        self.drl_agent.store_data(**kwargs)

    def config(self):
        return self.drl_agent.agent_config, self.drl_agent.memory_config, self.drl_agent.train_config

    def save(self, name):
        self.drl_agent.save(rl_model_path + f"\\{self.drl_name}\\{name}")

    def load(self, name):
        self.drl_agent.load(rl_model_path + f"\\{self.drl_name}\\{name}")

    def load_from_path(self,path):
        self.drl_agent.load(path)


class DRLRDWEnv(RdwEnv, ABC):

    def __init__(self):
        super().__init__()
        self.trajs = []

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, eval_turns=2):
        raise NotImplementedError

    def obtain_rand_p_initloc(self):
        p_scene_rect = np.array(self.p_scene.bounds[0].cir_rect)
        x_min, y_min = np.min(p_scene_rect, axis=0)
        x_max, y_max = np.max(p_scene_rect, axis=0)
        init_p_loc = None
        while init_p_loc is None:
            loc = [np.random.uniform(x_min + 30, x_max - 30), np.random.uniform(y_min + 30, y_max - 30)]
            if self.p_scene.poly_contour_safe.contains(Point(loc)):
                init_p_loc = loc
        return init_p_loc
