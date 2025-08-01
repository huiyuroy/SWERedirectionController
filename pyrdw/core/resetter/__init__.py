from pyrdw.core import *


class BaseResetter(BaseManager):

    def __init__(self):
        super().__init__()
        self.reset_spec = const_reset
        self.reset_type = 'base'
        self.reset_pre_state = 0
        self.reset_state = 0
        self.reset_angle = 0
        self.reset_scale = 1
        # 距离边界的最小距离，小于该距离触发reset
        self.reset_trigger_t = 20
        # 结束reset状态的最小角度误差，当前实际方向与目标方向小于该值时终止reset
        self.reset_terminal_t = 1
        self.reset_pred_t = 40
        self.reset_target_fwd = None
        self.reset_rest_angle = PI
        self.reset_start_vir_fwd = [0, 1]
        self.p_loc, self.p_fwd = np.array([0, 0]), np.array([0, 0])
        self.v_loc, self.v_fwd = np.array([0, 0]), np.array([0, 0])
        self.p_vel, self.p_rot = 0, 0
        self.reset_num = 0
        self.enable_draw = True

    def load_params(self):
        self.reset_trigger_t = self.reset_spec['reset_trigger_dis']
        self.reset_terminal_t = self.reset_spec['reset_finish_ang'] * DEG2RAD
        self.reset_pred_t = self.reset_spec['reset_pred_dis']

    def prepare(self):
        self.p_scene, self.v_scene = self.agent.p_scene, self.agent.v_scene

    def reset(self):
        self.reset_angle = 0
        self.reset_state = 0
        self.reset_pre_state = 0
        self.reset_target_fwd = None
        self.reset_rest_angle = PI
        self.reset_start_vir_fwd = [0, 1]
        self.p_loc, self.p_fwd = np.array([0, 0]), np.array([0, 0])
        self.v_loc, self.v_fwd = np.array([0, 0]), np.array([0, 0])
        self.p_vel, self.p_rot = 0, 0
        self.reset_num = 0

    def is_resetting(self):
        return self.reset_state

    def enable(self):
        self.reset_state = 1

    def disable(self):
        self.reset_state = 0

    def record(self):
        self.reset_pre_state = self.reset_state

    def trigger(self):
        if self.reset_state:
            self.reset_num += 1
            return 1
        else:
            if self.p_scene.poly_contour_safe.covers(Point(self.p_loc)):
                self.reset_state = 0
            else:
                nx_p_loc = self.p_loc + geo.norm_vec(self.p_fwd) * self.reset_pred_t * 2

                if (self.p_vel > 0  # 判断移动时，是否安全
                        and (not self.p_scene.poly_contour_safe.covers(Point(nx_p_loc))
                             or not self.p_scene.poly_contour.covers(LineString([self.p_loc, nx_p_loc])))):
                    self.reset_num += 1
                    self.reset_state = 1
                else:
                    self.reset_state = 0
            # tiling = self.agent.p_cur_tiling
            # if tiling.type and (tiling.sur_occu_safe or self.p_scene.poly_contour_safe.covers(Point(self.p_loc))):
            #     self.reset_state = 0
            # else:
            #     nx_p_loc = self.p_loc + geo.norm_vec(self.p_fwd) * self.reset_pred_t * 2
            #     if ((not self.p_scene.poly_contour_safe.covers(Point(nx_p_loc))
            #          or not self.p_scene.poly_contour.covers(LineString((self.p_loc, nx_p_loc))))
            #             and self.p_vel > 0):
            #         self.reset_num += 1
            #         self.reset_state = 1
            #     else:
            #         self.reset_state = 0
            return self.reset_state

    def start(self, **kwargs):
        self.calc_reset_target_fwd()
        self.reset_angle = geo.calc_angle_bet_vec(self.p_fwd, self.reset_target_fwd)
        if self.reset_angle == 0:
            self.reset_state = 0
            return
        self.reset_start_vir_fwd = self.v_fwd.copy()
        self.reset_rest_angle = self.reset_angle
        self.reset_scale = abs(PI_2 / self.reset_angle)

    @abstractmethod
    def calc_reset_target_fwd(self):
        """
        用于在启动重置时计算重置目标方向，自己的重置方法需要对这个进行重写

        Returns:
            None
         """
        raise NotImplementedError

    def update(self, **kwargs):
        self.reset_rest_angle = geo.calc_angle_bet_vec(self.p_fwd, self.reset_target_fwd)
        head_fwd = geo.norm_vec(geo.rot_vecs(self.reset_start_vir_fwd, -self.reset_rest_angle * self.reset_scale))
        if abs(self.reset_rest_angle) < self.reset_terminal_t:
            self.reset_state = 0
            move_fwd = head_fwd * self.p_vel
            self.reset_target_fwd = None
        else:
            self.reset_state = 1
            move_fwd = [0, 0]
        self.agent.v_cur_loc = self.v_loc + move_fwd
        self.agent.v_cur_fwd = head_fwd

    def render(self, wdn_obj, default_color):
        if self.reset_target_fwd is not None:
            wdn_obj.draw_phy_line(self.p_loc, geo.norm_vec(self.reset_target_fwd) * 150 + self.p_loc, 3, (255, 0, 0))

    def copy_target_manager(self, other_mg):
        if other_mg is not None:
            self.reset_trigger_t = other_mg.reset_trigger_t
            self.reset_terminal_t = other_mg.reset_terminal_t
        else:
            self.load_params()
