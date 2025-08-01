import numpy as np

from pyrdw import *

import pyrdw.lib.math as pymath
import pyrdw.lib.math.algebra as alg
import pyrdw.lib.math.geometry as geo
import pyrdw.lib.math.image as img

from pyrdw.core.constant import *
from pyrdw.vis.ui.base import RDWWindow

TIME_STEP = const_env['time_step']
GRID_WIDTH = const_env['tiling_width']

EPS = pymath.EPS
DEG2RAD = pymath.DEG2RAD
RAD2DEG = pymath.RAD2DEG
PI = pymath.PI
PI_2 = pymath.PI_2
PI_1_2 = pymath.PI_1_2
PI_1_4 = pymath.PI_1_4

REV_PI = 1 / pymath.PI
REV_PI_2 = 1 / pymath.PI_2
REV_PI_SQUARE = REV_PI ** 2

HUMAN_RADIUS = const_env['human_radius']
HUMAN_STEP = const_simu['human_step']
REV_HUMAN_STEP = 1 / HUMAN_STEP


class BaseManager(ABC):

    def __init__(self):
        self.agent = None
        self.time_step = const_env["time_step"]
        self.v_scene = None
        self.p_scene = None
        self.p_loc, self.p_fwd = np.array([0, 0]), np.array([0, 0])
        self.v_loc, self.v_fwd = np.array([0, 0]), np.array([0, 0])
        self.p_vel_vec = np.array([0, 0])
        self.p_vel, self.p_rot = 0, 0

    def setup_agent(self, agent):
        self.agent = agent

    def obtain_agent_scenes(self):
        self.p_scene, self.v_scene = self.agent.p_scene, self.agent.v_scene

    @abstractmethod
    def load_params(self):
        raise NotImplementedError

    @abstractmethod
    def prepare(self):
        """
        设置虚实空间及模拟行走路径后调用

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """

        Returns:

        """
        raise NotImplementedError

    def update_pv_states(self):
        self.p_loc, self.p_fwd = self.agent.p_cur_loc, self.agent.p_cur_fwd
        self.v_loc, self.v_fwd = self.agent.v_lst_loc, self.agent.v_lst_fwd
        self.p_vel, self.p_rot = self.agent.p_cur_vel, self.agent.p_cur_rot
        self.p_vel_vec = self.agent.p_vel_vec

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def render(self, wdn_obj: RDWWindow, default_color):
        raise NotImplementedError
