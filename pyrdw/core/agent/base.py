from pyrdw.core import *
from pyrdw.core.space.grid import Tiling
from pyrdw.lib.math.geometry import ConvexPoly


class BaseAgent(ABC):
    """
    base agent to contain all components of rdw methods, including gain controller, rdw controller, reset controller,
    input controller.
    """

    def __init__(self):
        self.name = 'base'
        # Gain Manager: manages all rdw gain parameters and must be called when rdw gain is required in the agent.
        self.gainer = None
        # Redirect Manager: manages rdw method parameters and must be invoked when rdw is used in the agent.
        self.rdwer = None
        # Reset Manager: manages reset method parameters and must be invoked when resets are used in the agent.
        self.resetter = None
        # Walking Manager: manages various parameters of simulated and actual walking.
        self.inputer = None
        # --------------------------Variables related to virtual and real scenes-----------------------------------
        self.v_scene, self.p_scene = None, None
        # User status data, pre: t_-2， lst: t_-1， cur: t_0,
        # Each line represents a single-moment state, organized as follows: position (x_l, y_l), orientation (x_r, y_r),
        # movement speed (v), and angular velocity (theta, with +/- indicating rotation direction).
        self.v_pre_loc, self.v_pre_fwd, self.v_pre_vel, self.v_pre_rot = np.array([0, 0]), np.array([0, 0]), 0, 0
        self.v_lst_loc, self.v_lst_fwd, self.v_lst_vel, self.v_lst_rot = np.array([0, 0]), np.array([0, 0]), 0, 0
        self.v_cur_loc, self.v_cur_fwd, self.v_cur_vel, self.v_cur_rot = np.array([0, 0]), np.array([0, 0]), 0, 0
        self.v_vel_vec = np.array([0, 0])
        self.v_tar_loc, self.v_tar_fwd = np.array([0, 0]), np.array([0, 0])
        self.p_pre_loc, self.p_pre_fwd, self.p_pre_vel, self.p_pre_rot = np.array([0, 0]), np.array([0, 0]), 0, 0
        self.p_lst_loc, self.p_lst_fwd, self.p_lst_vel, self.p_lst_rot = np.array([0, 0]), np.array([0, 0]), 0, 0
        self.p_cur_loc, self.p_cur_fwd, self.p_cur_vel, self.p_cur_rot = np.array([0, 0]), np.array([0, 0]), 0, 0
        self.p_vel_vec = np.array([0, 0])
        self.v_init_loc, self.v_init_fwd, self.v_init_vel, self.v_init_rot = np.array([0, 0]), np.array([0, 0]), 0, 0
        self.p_init_loc, self.p_init_fwd, self.p_init_vel, self.p_init_rot = np.array([0, 0]), np.array([0, 0]), 0, 0

        self.epi_step = 0  # current episode step, exclude reset procedure
        self.time_step = 0

        self.rdw_rate = 0

    def set_manager(self, manager, m_type=None):
        """
        set the specific manager for each component
        Args:
            manager:
            m_type:

        Returns:

        """
        if 'gain' in m_type:
            self.gainer = manager
            self.gainer.setup_agent(self)
        elif 'rdw' in m_type:
            self.rdwer = manager
            self.rdwer.setup_agent(self)
        elif 'reset' in m_type:
            self.resetter = manager
            self.resetter.setup_agent(self)
        elif 'input' in m_type:
            self.inputer = manager
            self.inputer.setup_agent(self)

    def set_scenes(self, v_scene=None, p_scene=None):
        """
        set virtual and physical scenes.

        Args:
            v_scene:
            p_scene:

        Returns:

        """
        if v_scene is not None:
            self.v_scene = v_scene
        if p_scene is not None:
            self.p_scene = p_scene
        self.gainer.obtain_agent_scenes()
        self.rdwer.obtain_agent_scenes()
        self.resetter.obtain_agent_scenes()
        self.inputer.obtain_agent_scenes()

    def prepare(self):
        """
        prepare offline setups that are necessary for post runtime calculations.
        Must be called before agent.reset() function.

        Returns:

        """
        self.gainer.prepare()
        self.rdwer.prepare()
        self.resetter.prepare()
        self.inputer.prepare()

    def reset(self):
        """
        reset all components of the agent. must be called before using the agent.

        Returns:

        """
        # Determine the user's virtual and real state (position and orientation) each time walking begins.
        # recommend to re-write this method if you are building a new agent type.
        self.epi_step = 0

        self.p_cur_loc, self.p_cur_fwd, self.v_tar_loc, self.v_tar_fwd = self.inputer.reset()
        self.p_pre_loc, self.p_pre_fwd = self.p_lst_loc, self.p_lst_fwd
        self.p_pre_vel, self.p_pre_rot = self.p_lst_vel, self.p_lst_rot
        self.p_lst_loc, self.p_lst_fwd = self.p_cur_loc, self.p_cur_fwd

        self.v_cur_loc, self.v_cur_fwd = self.v_tar_loc, self.v_tar_fwd
        self.v_pre_loc, self.v_pre_fwd = self.v_lst_loc, self.v_lst_fwd
        self.v_pre_vel, self.v_pre_rot = self.v_lst_vel, self.v_lst_rot
        self.v_lst_loc, self.v_lst_fwd = self.v_cur_loc, self.v_cur_fwd

        self.gainer.reset()
        self.rdwer.reset()
        self.resetter.reset()
        s = self.step_state()
        return s

    def step_early(self, **kwargs):
        """
        The agent follows three continuous methods to update itself.

        Order: step_early -> step -> step_late.

        I recommend to re-write this method if your agent needs additional manipulations before applying the
        rdw operations.

        """
        self.p_vel_vec = self.p_cur_loc - self.p_lst_loc
        self.p_cur_vel = alg.l2_norm(self.p_vel_vec)  # Walking speed
        self.p_cur_rot = geo.calc_angle_bet_vec(self.p_lst_fwd, self.p_cur_fwd)  # Rotation speed

        self.inputer.update_pv_states()
        self.gainer.update_pv_states()
        self.rdwer.update_pv_states()
        self.resetter.update_pv_states()

    def step(self, **kwargs):
        """
        Applying all necessary operations of rdw. Unless necessary, no changes are required.

        """
        if self.resetter.is_resetting():  # The agent is in the reset state.
            self.resetter.record()
            self.resetter.update(**kwargs)  # update reset controller
        else:  # otherwise
            self.resetter.record()
            self.resetter.disable()
            self.gainer.update(**kwargs)
            self.rdwer.update(**kwargs)  # update rdw controller
            self.rdwer.calc_vir_state()  # calculate the corresponding virtual state
            if self.resetter.trigger():  # Trigger the reset
                self.resetter.start(**kwargs)

    def step_late(self, **kwargs):
        """
        Some operations to apply after the update of rdw controller. For example, if you are creating a deep
        reinforcement learning environment, the reward can be calculated in this method.

        Returns:

        """
        # state update, must remain.
        self.v_pre_loc, self.v_pre_fwd = self.v_lst_loc, self.v_lst_fwd
        self.v_pre_vel, self.v_pre_rot = self.v_lst_vel, self.v_lst_rot
        self.v_lst_loc, self.v_lst_fwd = self.v_cur_loc, self.v_cur_fwd
        self.v_lst_vel, self.v_lst_rot = self.v_cur_vel, self.v_cur_rot

        self.p_pre_loc, self.p_pre_fwd = self.p_lst_loc, self.p_lst_fwd
        self.p_pre_vel, self.p_pre_rot = self.p_lst_vel, self.p_lst_rot
        self.p_lst_loc, self.p_lst_fwd = self.p_cur_loc, self.p_cur_fwd
        self.p_lst_vel, self.p_lst_rot = self.p_cur_vel, self.p_cur_rot
        # other implementation..........

    @abstractmethod
    def step_state(self):
        """
        Additional method to reorganize the output states.

        Returns:

        """
        raise NotImplementedError

    def state_refresh(self, **kwargs):
        """
        update all user states. Called after step_late.

        Returns:
            s_next, done, truncated, info
        """
        self.p_cur_loc, self.p_cur_fwd, self.v_tar_loc, self.v_tar_fwd, done = self.inputer.update()
        # -------------------------------------
        # time of a frame, sync with inputer
        self.time_step = self.inputer.time_step
        self.gainer.time_step = self.time_step
        self.rdwer.time_step = self.time_step
        self.resetter.time_step = self.time_step
        # --------------------------------------
        if not done:
            self.step_early()
            self.step()
            self.step_late()
            s = self.step_state()
        else:
            s = None
        return s, done, False, None

    def state_log(self):
        reset_info = []
        self.rdw_rate = 0
        if not self.resetter.reset_state:
            reset_state = 0
            reset_type = 'none'
            if not self.resetter.reset_pre_state:
                self.rdw_rate = abs(abs(self.p_cur_rot) - abs(self.v_cur_rot))
        else:  # resetting
            reset_state = 1
            reset_type = self.resetter.reset_type
            if not self.resetter.reset_pre_state:
                reset_info.extend(self.p_cur_loc)
                reset_info.extend(self.resetter.reset_target_fwd)
                reset_info.append(self.resetter.reset_angle)

        step_state = {
            't': self.epi_step,
            'pwv': self.p_cur_vel,
            'prv': self.p_cur_rot,
            'p_loc': self.p_cur_loc,
            'p_fwd': self.p_cur_fwd,
            'v_loc': self.v_cur_loc,
            'v_fwd': self.v_cur_fwd,
            'vwv': self.v_cur_vel,
            'vrv': self.v_cur_rot,
            'rdw_r': self.rdw_rate,
            'rs_s': reset_state,
            'rs_t': reset_type,
            'rs_n': self.resetter.reset_num,
            'rs_info': reset_info
        }
        return step_state

    @abstractmethod
    def render(self, wdn_obj: RDWWindow, default_color):
        raise NotImplementedError


class GeneralRdwAgent(BaseAgent):
    """
    A general rdw agent used to represent a simu/live user in the environment.
    """

    def __init__(self):
        super().__init__()
        self.v_cur_tiling: Tiling = None
        self.p_cur_tiling: Tiling = None
        self.v_cur_conv: ConvexPoly = None
        self.p_cur_conv: ConvexPoly = None
        self.enable_phy_discrete = True
        self.enable_vir_discrete = True
        self.enable_render = False

    def reset(self):
        self.epi_step = 0
        self.p_cur_loc, self.p_cur_fwd, self.v_tar_loc, self.v_tar_fwd = self.inputer.reset()
        self.p_pre_loc, self.p_pre_fwd = self.p_lst_loc, self.p_lst_fwd
        self.p_pre_vel, self.p_pre_rot = self.p_lst_vel, self.p_lst_rot
        self.p_lst_loc, self.p_lst_fwd = self.p_cur_loc, self.p_cur_fwd

        self.v_cur_loc, self.v_cur_fwd = self.v_tar_loc, self.v_tar_fwd
        self.v_pre_loc, self.v_pre_fwd = self.v_lst_loc, self.v_lst_fwd
        self.v_pre_vel, self.v_pre_rot = self.v_lst_vel, self.v_lst_rot
        self.v_lst_loc, self.v_lst_fwd = self.v_cur_loc, self.v_cur_fwd
        self.scene_discrete_update()

        self.gainer.reset()
        self.rdwer.reset()
        self.resetter.reset()
        s = self.step_state()
        return s

    def step_early(self, **kwargs):
        super().step_early()
        self.scene_discrete_update()

    def scene_discrete_update(self):
        if self.enable_phy_discrete:
            self.p_cur_tiling, self.p_cur_conv = self.p_scene.calc_located_tiling_conv(self.p_cur_loc)
        if self.enable_vir_discrete:
            self.v_cur_tiling, self.v_cur_conv = self.v_scene.calc_located_tiling_conv(self.v_cur_loc)

    def step_state(self):
        p_state = np.concatenate((self.p_lst_loc, [0], self.p_lst_fwd, [0]))
        v_state = np.concatenate((self.v_lst_loc, [0], self.v_lst_fwd, [0]))
        e_v = max(self.v_scene.max_size) * 0.5
        e_p = max(self.p_scene.max_size) * 0.5
        state = np.concatenate((p_state, v_state, [e_p, e_v])).tolist()
        return state.copy()

    def render(self, wdn_obj: RDWWindow, default_color):
        wdn_obj.draw_vir_circle(self.v_cur_loc, 20, default_color)
        wdn_obj.draw_vir_line(self.v_cur_loc, geo.norm_vec(self.v_cur_fwd) * 100 + self.v_cur_loc, 2, default_color)
        wdn_obj.draw_vir_circle(self.v_tar_loc, 2, default_color)
        wdn_obj.draw_vir_line_bg(self.v_cur_loc, self.v_lst_loc, 2, default_color)

        wdn_obj.draw_phy_circle(self.p_cur_loc, 20, default_color)
        wdn_obj.draw_phy_line(self.p_cur_loc, geo.norm_vec(self.p_cur_fwd) * 100 + self.p_cur_loc, 2, default_color)
        wdn_obj.draw_phy_line_bg(self.p_cur_loc, self.p_lst_loc, 2, default_color)
