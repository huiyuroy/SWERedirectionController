from pyrdw.core import *


class BaseInputer(BaseManager):

    def __init__(self):
        super().__init__()
        self.simu_spec = const_simu
        self.time_counter = 0
        self.time_reset_counter = 0
        self.move_counter = 1
        self.time_range = 1
        # 模拟用户的初始状态，前两位-位置，后两位-朝向
        self.init_p_state = [0, 0, 0, 0]
        self.init_v_state = [0, 0, 0, 0]
        self.reset_state = None

    def setup_agent(self, env):
        super().setup_agent(env)

    def load_params(self):
        pass

    def prepare(self):
        self.p_scene, self.v_scene = self.agent.p_scene, self.agent.v_scene

    def set_time_step(self, t):
        self.time_step = t

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError

    def set_vir_init_state(self, loc, fwd):
        self.init_v_state = loc
        self.init_v_state.extend(fwd)

    def set_phy_init_state(self, loc, fwd, rand=False):
        self.init_p_state = [0, 0]
        self.init_p_state.extend(fwd)


class SimuTrajectoryInputer(BaseInputer):
    """
    模拟在指定路径上行走，模拟用户行动时，尽可能加速到预定最大速度（行走速度、转向速度）上行动
    """

    def __init__(self):
        super().__init__()
        # 模拟用户移动速度的范围[min,max], 模拟用户旋转速度的范围[min,max]
        self.mv_vel_range, self.rot_vel_range = [0, 0], [0, 0]
        # 模拟用户的默认行走速度[0]和旋转速度[1]
        self.norm_phy_vels = [0, 0]
        # 模拟用户在虚拟空间与虚拟目标的最小距离[0]和角度朝向[1]，都小于给定值时，认为用户已经到达虚拟目标位置，可进行下一次目标选择
        self.end_motion_thr = [0, 0]
        # 模拟用户当前速度，[0]是行走速度，[1]是旋转速度(+/-表示旋转方向)
        self.cur_phy_vels = [0, 0]
        self.mv_acc = 0  # 行走加速度
        self.mr_acc = 0  # 转向加速度
        self.acc_rate = 0.25  # 4帧加速到最大速度
        self.predict_frame = 4  # 预测4帧后的状态
        # 模拟用户当前对应的虚拟目标，包括目标位置，目标朝向
        self.vir_tar_loc = np.array([0, 0])
        self.vir_tar_fwd = np.array([0, 0])
        # 模拟用户是否在距离[0]和朝向[1]上到达当前指定虚拟目标，其中朝向指用户正方向与其和虚拟目标间方向是否配准
        self.tar_done = [False, False]
        # 记录模拟用户当前虚拟状态与目标状态间的偏差度，[0][1]当前距离和方向偏差度，[2][3]前一帧偏差度
        self.v_offset2tar = [0, 0, 0, 0]
        # 随机选择物理空间起点
        self.rand_init_p_state = True
        self.cur_traj = None
        self.cur_tars = np.zeros((10, 2))
        self.cur_tar_idx = 0
        self.cur_tar_nums = 0
        self.cur_step = 0
        self.walk_finished = False
        self.delta_time = None

    def load_params(self):
        self.mv_vel_range = np.array(self.simu_spec["walk_vel_range"][0:2])
        self.rot_vel_range = np.array(self.simu_spec["rotate_vel_range"][0:2]) * DEG2RAD
        self.norm_phy_vels = np.array([self.simu_spec["norm_walk_vel"][0], self.simu_spec["norm_rot_vel"][0]])
        self.end_motion_thr = np.array(self.simu_spec["end_motion_thresholds"][0:2])
        self.end_motion_thr[1] = self.end_motion_thr[1] * DEG2RAD

    def set_params(self,
                   move_range: np.ndarray = None,
                   rot_range: np.ndarray = None):
        if move_range is not None:
            self.mv_vel_range = move_range
        if rot_range:
            self.rot_vel_range = rot_range

    def set_trajectory(self, traj):
        self.cur_traj = traj

    def set_phy_init_state(self, loc, fwd, rand=False):
        self.rand_init_p_state = rand
        if not self.rand_init_p_state:
            self.init_p_state = np.array([loc[0], loc[1], fwd[0], fwd[1]])
        else:
            pot_tilings = []
            for t in self.p_scene.tilings:
                if t.type:
                    can_add = True
                    for n_id in t.nei_ids:
                        if not self.p_scene.tilings[n_id].type:
                            can_add = False
                    if can_add:
                        pot_tilings.append(t)
            select_t = pot_tilings[np.random.choice(len(pot_tilings))]
            x = np.random.rand() * (select_t.rect[1][0] - select_t.rect[0][0]) + select_t.rect[0][0]
            y = np.random.rand() * (select_t.rect[2][1] - select_t.rect[0][1]) + select_t.rect[0][1]
            self.init_p_state = [x, y]
            self.init_p_state.extend(geo.rot_vecs(np.array([0, 1]), np.random.uniform(-0.99, 0.99) * PI).tolist())

    def reset(self):
        """
        以指定record路径重置walker,每次重置后选择下一条路径链

        Returns: 初始状态
        """
        self.mv_acc = self.mv_vel_range[1] * self.acc_rate
        self.mr_acc = self.rot_vel_range[1] * self.acc_rate
        self.cur_tars = self.cur_traj.walkable()
        self.cur_tar_nums = len(self.cur_tars)
        self.cur_tar_idx = 0
        self.cur_step = 0
        self.tar_done = [False, False]
        self.cur_phy_vels = [0, 0]
        self.vir_tar_loc = self.cur_tars[0][:2]
        self.vir_tar_fwd = np.array(self.init_v_state[2:4])
        self.p_loc = np.array(self.init_p_state[:2])
        self.p_fwd = np.array(self.init_p_state[2:4])

        return self.p_loc, self.p_fwd, self.vir_tar_loc, self.vir_tar_fwd

    def update(self, **kwargs):
        self.delta_time = 1 / self.time_step
        p_loc, p_fwd = self.agent.p_lst_loc, self.agent.p_lst_fwd
        v_loc, v_fwd = self.agent.v_lst_loc, self.agent.v_lst_fwd
        vp_loc, vp_fwd, = self.agent.v_pre_loc, self.agent.v_pre_fwd
        min_mv, max_mv = self.mv_vel_range
        min_rv, max_rv = self.rot_vel_range

        self.walk_finished = False
        self.v_offset2tar[2:] = self.v_offset2tar[0:2]
        self.v_offset2tar[0] = alg.l2_norm(v_loc - self.vir_tar_loc)
        self.v_offset2tar[1] = geo.calc_angle_bet_vec(v_fwd, self.vir_tar_fwd)
        self.cur_step += 1
        if self.cur_tar_idx >= self.cur_tar_nums - 1:
            self.walk_finished = True
            return p_loc, p_fwd, self.vir_tar_loc, self.vir_tar_fwd, self.walk_finished

        if not self.agent.resetter.reset_state:  # 不在reset
            self.reset_state = self.agent.resetter.reset_state
            self.time_reset_counter = 0
            abs_fwd2tar = abs(self.v_offset2tar[1])
            if self.tar_done[1]:  # 完成了朝向配准
                if self.v_offset2tar[0] <= self.end_motion_thr[0]:  # 完成朝向配准加位置配准，此时选择下一个目标位置
                    self.cur_tar_idx += 1
                    self.vir_tar_loc = self.cur_tars[self.cur_tar_idx][:2]
                    self.vir_tar_fwd = self.vir_tar_loc - v_loc
                    self.tar_done = [False, False]
                    pdir = alg.sign(geo.calc_angle_bet_vec(v_fwd, self.vir_tar_fwd))
                    self.cur_phy_vels = [0, pdir * self.mr_acc]
                elif math.sin(abs_fwd2tar) * alg.l2_norm(self.vir_tar_fwd) > self.end_motion_thr[0]:
                    self.tar_done[1] = False
                    self.cur_phy_vels = [0, 0]
                else:  # 没完成位置配准，还需让用户再向前走
                    mv = self.cur_phy_vels[0] + self.mv_acc
                    self.cur_phy_vels = [alg.clamp(mv, 0, max_mv),
                                         geo.calc_angle_bet_vec(v_fwd, vp_fwd) * self.delta_time]
                    if self.cur_phy_vels[0] * self.time_step > self.v_offset2tar[0]:
                        self.cur_phy_vels[0] = alg.clamp(self.v_offset2tar[0] * self.delta_time, 0, max_mv)
            else:  # 没完成位置配准，还需让用户调整朝向
                self.cur_phy_vels[0] = 0
                if abs_fwd2tar > self.end_motion_thr[1]:
                    mr = self.cur_phy_vels[1] + alg.sign(self.v_offset2tar[1]) * self.mr_acc
                    self.cur_phy_vels = [0, alg.clamp(mr, -max_rv, max_rv)]
                    if abs(self.cur_phy_vels[1] * self.time_step) > abs(self.v_offset2tar[1]):
                        self.cur_phy_vels[1] = self.v_offset2tar[1] * self.delta_time
                elif math.sin(abs_fwd2tar) * alg.l2_norm(self.vir_tar_fwd) <= self.end_motion_thr[0]:
                    self.tar_done = [False, True]
                    self.cur_phy_vels[1] = 0
                else:
                    self.cur_phy_vels[1] = self.v_offset2tar[1] * self.delta_time * 0.5
            p_next_fwd = geo.norm_vec(geo.rot_vecs(p_fwd, self.cur_phy_vels[1] * self.time_step))
            p_next_loc = p_loc + p_next_fwd * (self.cur_phy_vels[0] * self.time_step)
        else:  # 需要进行reset
            expected_dir = 1 if self.agent.resetter.reset_rest_angle > 0 else -1
            self.tar_done = [False, False]
            self.cur_phy_vels = [0, expected_dir * max_rv * 2]  # 如果重置则旋转速度是普通速度的二倍
            abs_reset = abs(self.agent.resetter.reset_rest_angle)
            if abs_reset <= abs(self.agent.resetter.reset_angle * 0.1) or abs_reset < 5 * DEG2RAD:
                self.cur_phy_vels[1] = expected_dir * abs(self.agent.resetter.reset_rest_angle * self.delta_time)
            p_next_fwd = geo.norm_vec(geo.rot_vecs(p_fwd, self.cur_phy_vels[1] * self.time_step))
            p_next_loc = p_loc + p_next_fwd * self.cur_phy_vels[0] * self.time_step
            self.cur_phy_vels[1] = 0

        return p_next_loc, p_next_fwd, self.vir_tar_loc, self.vir_tar_fwd, self.walk_finished

    def render(self, wdn_obj, default_color):
        pass

    def copy_target_manager(self, other_mg):
        if other_mg is not None:
            self.mv_vel_range = np.array(other_mg.mv_vel_range).copy().tolist()
            self.rot_vel_range = np.array(other_mg.rot_vel_range).copy().tolist()
            self.norm_phy_vels = np.array(other_mg.norm_phy_vels).copy().tolist()
            self.end_motion_thr = np.array(other_mg.end_motion_thr).copy().tolist()
        else:
            self.load_params()


class LiveInputer(SimuTrajectoryInputer):

    def __init__(self):
        super().__init__()

    def reset(self):
        """
               以指定record路径重置walker,每次重置后选择下一条路径链

               Returns: 初始状态
               """

        self.v_loc = np.array(self.init_v_state[:2])
        self.v_fwd = np.array(self.init_v_state[2:4])

        self.p_loc = np.array(self.init_p_state[:2])
        self.p_fwd = np.array(self.init_p_state[2:4])
        return self.p_loc, self.p_fwd, self.v_loc, self.v_fwd

    def set_phy_state(self, loc, fwd):
        self.p_loc = loc
        self.p_fwd = fwd

    def update(self, **kwargs):
        self.delta_time = 1 / self.time_step

        return self.p_loc, self.p_fwd, self.v_loc, self.v_fwd, False

    def render(self, wdn_obj, default_color):
        wdn_obj.draw_vir_circle(self.v_loc, 10, default_color)
        wdn_obj.draw_vir_line(self.v_loc, geo.norm_vec(self.v_fwd) * 100 + self.v_loc, 2, default_color)
        wdn_obj.draw_phy_circle(self.p_loc, 20, default_color)
        wdn_obj.draw_phy_line(self.p_loc, geo.norm_vec(self.p_fwd) * 100 + self.p_loc, 2, default_color)
