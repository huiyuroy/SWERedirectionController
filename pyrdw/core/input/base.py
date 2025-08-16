
from pyrdw.core import *


class BaseInputer(BaseManager):

    def __init__(self):
        super().__init__()
        self.simu_spec = const_simu
        self.time_counter = 0
        self.time_reset_counter = 0
        self.move_counter = 1
        self.time_range = 1
        # Initial state of simulated user, first two - position, last two - orientation
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
    Simulate walking on a specified trajectory, when simulating user actions, try to accelerate to the predetermined maximum speed (walking speed, rotation speed)
    """

    def __init__(self):
        super().__init__()
        # Range of simulated user movement speed [min,max], range of simulated user rotation speed [min,max]
        self.mv_vel_range, self.rot_vel_range = [0, 0], [0, 0]
        # Default walking speed [0] and rotation speed [1] of simulated user
        self.norm_phy_vels = [0, 0]
        # Minimum distance [0] and angular orientation [1] between simulated user in virtual space and virtual target, when both are less than the given value, it is considered that the user has reached the virtual target position and can proceed to the next target selection
        self.end_motion_thr = [0, 0]
        # Current speed of simulated user, [0] is walking speed, [1] is rotation speed (+/- indicates rotation direction)
        self.cur_phy_vels = [0, 0]
        self.mv_acc = 0  # Walking acceleration
        self.mr_acc = 0  # Rotation acceleration
        self.acc_rate = 0.25  # Accelerate to maximum speed in 4 frames
        self.predict_frame = 4  # Predict state after 4 frames
        # Current virtual target corresponding to simulated user, including target position and target orientation
        self.vir_tar_loc = np.array([0, 0])
        self.vir_tar_fwd = np.array([0, 0])
        # Whether simulated user reaches current specified virtual target in terms of distance [0] and orientation [1], where orientation refers to whether the user's forward direction is aligned with the direction between the user and the virtual target
        self.tar_done = [False, False]
        # Record the deviation degree between current virtual state and target state of simulated user, [0][1] current distance and direction deviation degree, [2][3] deviation degree of previous frame
        self.v_offset2tar = [0, 0, 0, 0]
        # Randomly select physical space starting point
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
        Reset walker with specified record path, select next path chain after each reset

        Returns: Initial state
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

        if not self.agent.resetter.reset_state:  # Not in reset
            self.reset_state = self.agent.resetter.reset_state
            self.time_reset_counter = 0
            abs_fwd2tar = abs(self.v_offset2tar[1])
            if self.tar_done[1]:  # Orientation alignment completed
                if self.v_offset2tar[0] <= self.end_motion_thr[0]:  # Orientation alignment plus position alignment completed, select next target position
                    self.cur_tar_idx += 1
                    self.vir_tar_loc = self.cur_tars[self.cur_tar_idx][:2]
                    self.vir_tar_fwd = self.vir_tar_loc - v_loc
                    self.tar_done = [False, False]
                    pdir = alg.sign(geo.calc_angle_bet_vec(v_fwd, self.vir_tar_fwd))
                    self.cur_phy_vels = [0, pdir * self.mr_acc]
                elif math.sin(abs_fwd2tar) * alg.l2_norm(self.vir_tar_fwd) > self.end_motion_thr[0]:
                    self.tar_done[1] = False
                    self.cur_phy_vels = [0, 0]
                else:  # Position alignment not completed, need to let user move forward
                    mv = self.cur_phy_vels[0] + self.mv_acc
                    self.cur_phy_vels = [alg.clamp(mv, 0, max_mv),
                                         geo.calc_angle_bet_vec(v_fwd, vp_fwd) * self.delta_time]
                    if self.cur_phy_vels[0] * self.time_step > self.v_offset2tar[0]:
                        self.cur_phy_vels[0] = alg.clamp(self.v_offset2tar[0] * self.delta_time, 0, max_mv)
            else:  # Position alignment not completed, need to let user adjust orientation
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
        else:  # Need to perform reset
            expected_dir = 1 if self.agent.resetter.reset_rest_angle > 0 else -1
            self.tar_done = [False, False]
            self.cur_phy_vels = [0, expected_dir * max_rv * 2]  # If resetting, rotation speed is twice the normal speed
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
            Reset walker with specified record path, select next path chain after each reset

            Returns: Initial state
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