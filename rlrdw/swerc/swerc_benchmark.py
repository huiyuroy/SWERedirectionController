"""

RL-RDW based on Navigation Compatibility with Auxiliary Task Learning

"""
import time

import numpy as np
import pygame
import pygame.gfxdraw

from drl_baseline.util import action_mapping
from pyrdw.core import GRID_WIDTH, HUMAN_RADIUS, TIME_STEP
from pyrdw.core.gain.base import SimpleGainManager
from pyrdw.core.input.base import SimuTrajectoryInputer, LiveInputer
from pyrdw.core.rdwer.reactive import NoRdwManager
from pyrdw.core.resetter import BaseResetter
from pyrdw.vis.ui.base import RDWWindow
from rlrdw import *
from rlrdw.swerc.config import swerc_state_dim, swerc_img_state_dim, swerc_action_repeat, swerc_state_seq, lidar_num, \
    swerc_max_v_dis
from rlrdw.swerc.ppo_policy import SWERCPPOAlg

MAX_VIR_WIDTH = 500  # refer to swerc network can deal with virtual spaces with maximal size of 50m x 50m. (each pixel -> a tiling)
MAX_PHY_WIDTH = 200  # refer to swerc network can deal with physical spaces with maximal size of 20m x 20m.
SHRUNK_SIZE = swerc_img_state_dim[1:]
# 设置颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class SWERCLidar:

    def __init__(self, fwd, contour, bounding_box):
        self.lidar = fwd
        self.lidar_contour = contour
        self.lidar_sur_box = bounding_box


class SWERCDiscreteScene(DiscreteScene):

    def __init__(self):
        super().__init__()

    def calc_swe_energy(self,sample_num=90):
        """
        Spatial walkability entropy

        Args:
            sample_num:
            free_comp:
            block_comp:
            safe_comp:

        Returns:

        """
        flat_sig = 0.1
        t_free_area = self.poly_contour.area
        w, h = self.tilings_shape
        swe_weights = np.zeros((h, w))
        sample_fwds = [geo.rot_vecs((0, 1), theta) for theta in np.linspace(0, np.pi * 2, sample_num, endpoint=False)]
        vec_len = alg.l2_norm_square
        max_trv_dis = 0

        for t_id, t in enumerate(self.tilings):
            setattr(t, 'swe_weight', 0)
            setattr(t, 'swe_grad', 0)
            # setattr(t, 'arl_dist_atte', 0)
            r, c = t.mat_loc
            if t.type:
                dis_prob_dist = []
                block_prob_dist = []
                ray_inters = []
                for sfwd in sample_fwds:
                    i_inter, i_dis = geo.calc_ray_poly_intersection(t.center, sfwd, t.vis_poly[0])
                    ray_inters.append(i_inter)
                    if i_dis > max_trv_dis:
                        max_trv_dis = i_dis
                for ray_i in range(sample_num):
                    s1 = ray_inters[ray_i]
                    if ray_i < sample_num - 1:
                        s2 = ray_inters[ray_i + 1]
                    else:
                        s2 = ray_inters[0]
                    pa = Polygon(shell=[t.center, s1, s2]).area
                    dis_prob_dist.append(pa / t.vis_poly[0].area)
                    block_prob_dist.append(pa / t_free_area)
                dis_prob_dist = np.array(dis_prob_dist)
                block_prob_dist = np.array(block_prob_dist)
                # travel reliability entropy
                tr_entropy = -np.sum(dis_prob_dist * np.log2(dis_prob_dist))
                # block degree entropy
                bl_entropy = -np.sum(block_prob_dist * np.log2(block_prob_dist))
                dist_atte = np.exp(-flat_sig / vec_len((t.nearst_obs_pos - t.center) * 0.01))

                t.swe_weight = tr_entropy * bl_entropy * dist_atte
            swe_weights[r, c] = t.swe_weight
            print('\r{} swerc energy precomputation: {:.2f}%'.format(self.name, t_id / len(self.tilings) * 100), end="")
        w_min = swe_weights.min()
        w_max = swe_weights.max()
        swe_weights = (swe_weights - w_min) / (w_max - w_min)
        swe_weights = img.blur_image(swe_weights, h, w)
        swe_grads = img.calc_img_grad(swe_weights, h, w)
        for i in range(h):
            for j in range(w):
                tiling = self.tilings[i * w + j]
                tiling.swe_weight = swe_weights[i, j]
                tiling.swe_grad = swe_grads[i, j]
        setattr(self, 'swe_weights', swe_weights)
        setattr(self, 'swe_grads', swe_grads)
        setattr(self, 'swe_max_tr_dis', max_trv_dis)
        setattr(self, 'swe_sum_weights', swe_weights.sum())

        sur_free_swe_weights = np.zeros((h, w))
        for idx, tiling in enumerate(self.tilings):
            r, c = tiling.mat_loc
            sur_tids, sur_radius = self.__spread2surround(tiling)
            setattr(tiling, 'sur_free_tids', sur_tids)
            setattr(tiling, 'sur_free_radius', sur_radius)
            sur_free_swe = 0
            for sur_id_c, sur_id_r in sur_tids:
                tid = sur_id_r * w + sur_id_c
                sur_free_swe += self.tilings[tid].swe_weight
            sur_free_swe_weights[r, c] = sur_free_swe
            setattr(tiling, 'sur_free_swe', sur_free_swe)
            print('\r{} swerc region precomputation: {:.2f}%'.format(self.name, idx / len(self.tilings) * 100), end="")
        setattr(self, 'sur_free_swe_weights', sur_free_swe_weights)
        print('\r{} swe energy precomputation: done'.format(self.name))

    def __spread2surround(self, tiling):
        diff_radius = geo.calc_point_mindis2bound(tiling.center, self.bounds)
        if diff_radius is not None and diff_radius > 0:
            spread_depth = int(diff_radius // self.tiling_w)
            r, c = tiling.mat_loc
            sur1, sur2 = self.calc_tiling_diffusion((c, r), (0, 1), 360, spread_depth, 0)
            sur_tiling_ids = sur1 + sur2
            return sur_tiling_ids, diff_radius
        else:
            return [], 0


class SWERCGainManager(SimpleGainManager):

    def __init__(self):
        super().__init__()
        self.g_min = (0.86, 0.8, -0.045)  # gt,gr,gc
        self.g_max = (1.26, 1.49, 0.045)  # gt,gr,gc
        self.g_values = [1, 1, 0]  # gt,gr,gc

    def reset(self):
        super().reset()
        self.g_values = [1, 1, 0]

    def update(self, **kwargs):
        """
        Input all actions, mapping to all rdw gains.

        Args:
            **kwargs:
                a_t [-1, 1] -> gt,
                a_r [-1, 1] -> gr,
                a_c [-1, 1] -> gc
        Returns:

        """
        if len(kwargs) > 0:  # new action comes in, need update g_values.
            a = kwargs['at'], kwargs['ar'], kwargs['ac']
            self.g_values = tuple(map(action_mapping, a, self.g_min, self.g_max))
            # print(kwargs['ac'], self.g_values[2])
        return self.g_values


class SWERCRDWManager(NoRdwManager):

    def __init__(self):
        super().__init__()
        self.mov_rot = 0

    def reset(self):
        super().reset()
        self.enable_rdw = True

    def calc_vir_state(self):
        gainer = self.agent.gainer
        p_fwd_vec, p_fwd_vel, p_ver_vec, p_ver_vel = self.p_vel_decompose()

        if self.enable_rdw:
            # 当前正在行走
            if p_fwd_vel > self.steer_vel_dt * self.time_step:
                self.mov_rot = 0.01 * p_fwd_vel * gainer.g_values[2]
                vir_rot_vel = self.p_rot + self.mov_rot  # 基于行走速度的方向偏转
            else:
                vir_rot_vel = self.p_rot * gainer.g_values[1]

            vir_fwd_vel = p_fwd_vel * gainer.g_values[0]
            vir_ver_vel = p_ver_vel
        else:
            vir_rot_vel = self.p_rot
            vir_fwd_vel = p_fwd_vel
            vir_ver_vel = p_ver_vel

        vir_fwd_vec_norm = geo.norm_vec(geo.rot_vecs(self.v_fwd, vir_rot_vel))
        vir_fwd_vec = vir_fwd_vec_norm * vir_fwd_vel
        vir_ver_vec = geo.rot_vecs(vir_fwd_vec_norm, -alg.sign(np.cross(p_fwd_vec, p_ver_vec)) * PI_1_2) * vir_ver_vel
        self.agent.v_vel_vec = vir_fwd_vec + vir_ver_vec
        self.agent.v_cur_loc = self.v_loc + self.agent.v_vel_vec
        self.agent.v_cur_fwd = vir_fwd_vec_norm
        self.agent.v_cur_vel = alg.l2_norm(self.agent.v_vel_vec)
        self.agent.v_cur_rot = vir_rot_vel

    def render(self, wdn_obj, default_color):
        pass


class SWERCResetter(BaseResetter):
    def __init__(self, r_type='normal'):
        super().__init__()
        self.r_type = r_type
        self.p_scene_diagonal_inv = 0
        self.sample_num = 630
        self.h_sample_num = int(self.sample_num / 2)
        self.sample_rot_step = 2 * PI / self.sample_num
        self.sample_rots = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        self.sample_rots_fwds = tuple([geo.rot_vecs(np.array([1, 0]), rot) for rot in self.sample_rots])
        self.sample_rots_atte = (2 * np.cos(self.sample_rots) + 3) / 5
        self.sample_rots_atte_sum = self.sample_rots_atte.sum()

        self.sample_dis_step = 20
        self.sample_render_p = []
        self.target_steer_tiling = None
        self.target_steer_gids = []
        self.target_steer_poly = None

    def prepare(self):
        super().prepare()
        c_rect = np.array(self.p_scene.bounds[0].cir_rect)
        self.p_scene_diagonal_inv = 1 / np.linalg.norm(np.max(c_rect, axis=0) - np.min(c_rect, axis=0))

    def calc_reset_target_fwd(self):
        """
        currently equals to r21
        新思路：
        重置方向熵最大（左右周围能量分布最平均）
        Returns:

        """
        v_cur_tiling, _ = self.v_scene.calc_located_tiling_conv(self.v_loc)
        p_cur_tiling, _ = self.p_scene.calc_located_tiling_conv(self.p_loc)
        p_vis_tris, p_grids = self.p_scene.update_visibility(self.p_loc, fwd=(0, 1),
                                                             grids_comp=True,
                                                             realtime_comp=False)
        all_simples_ids = list(gid for gids in p_grids for gid in gids)
        if self.r_type == 'normal':
            best_eng = -1
            best_fwd = geo.rot_vecs(self.p_fwd, PI)
            self.target_steer_tiling = p_cur_tiling
            for gid in all_simples_ids:
                g_tiling = self.p_scene.tilings[gid]
                g_point = g_tiling.center
                pot_fwd = geo.norm_vec(g_point - self.p_loc)
                nx_p_loc = self.p_loc + pot_fwd * self.reset_pred_t
                if (self.p_scene.poly_contour_safe.covers(Point(nx_p_loc)) and
                        self.p_scene.poly_contour.covers(LineString((self.p_loc, nx_p_loc)))):
                    sur_free_swe = g_tiling.sur_free_swe  # / (alg.l2_norm(g_point - self.p_loc) * 0.01)
                    if sur_free_swe > best_eng:
                        best_eng = sur_free_swe
                        best_fwd = pot_fwd
                        self.target_steer_tiling = g_tiling

            self.reset_target_fwd = best_fwd

        if self.r_type == 'complex':
            best_eng = -1
            best_fwd = geo.rot_vecs(self.p_fwd, PI)
            sample_rot_eng = []
            sample_rot_dis = []
            for fwd in self.sample_rots_fwds:
                nx_p_loc = self.p_loc + fwd * self.reset_pred_t
                if (self.p_scene.poly_contour_safe.covers(Point(nx_p_loc)) and
                        self.p_scene.poly_contour.covers(LineString((self.p_loc, nx_p_loc)))):
                    _, g_dis = geo.calc_ray_poly_intersection(self.p_loc, fwd, self.p_scene.poly_contour)
                    sample_dis_num = np.ceil(g_dis / self.sample_dis_step)
                    sample_step = g_dis / sample_dis_num
                    g_eng = 0
                    for dis in np.arange(sample_dis_num) * sample_step:
                        fwd_samplep = dis * fwd + self.p_loc
                        eng, _ = self.p_scene.interpolate_tiling_weight_grad(fwd_samplep,
                                                                             self.p_scene.sur_free_swe_weights)
                        g_eng += eng  # * (1 - dis * self.p_scene_diagonal_inv)
                    sample_rot_eng.append(g_eng)
                    sample_rot_dis.append(g_dis)
                else:
                    sample_rot_eng.append(0)
                    sample_rot_dis.append(0)
            len_s = len(sample_rot_eng)
            for fwd_i, f_eng in enumerate(sample_rot_eng):
                for fi in (np.arange(-60, 61) + fwd_i) % len_s:
                    f_eng += sample_rot_eng[fi]
                f_eng *= sample_rot_dis[fwd_i] * self.p_scene_diagonal_inv
                if f_eng > best_eng:
                    best_eng = f_eng
                    best_fwd = self.sample_rots_fwds[fwd_i]

            self.reset_target_fwd = best_fwd

    def render(self, wdn_obj, default_color):
        if self.reset_target_fwd is not None:
            wdn_obj.draw_phy_line(self.p_loc, geo.norm_vec(self.reset_target_fwd) * 500 + self.p_loc, 2,
                                  (255, 0, 0))

            if self.r_type == 'normal':
                wdn_obj.draw_phy_circle(self.target_steer_tiling.center,
                                        self.target_steer_tiling.sur_free_radius, (255, 0, 0))


class SWERCRDWAgent(DRLRDWAgent):
    """
    Spatial Walkability Entropy Redirected Controller
    """

    def __init__(self,
                 input_type='traj',
                 reset_type='normal'):
        """

        Args:
            enable_entropy: whether use the Spatial Walkability Entropy or not when calculating reward
            enable_align: whether use the navigation alignment or not when calculating reward
            input_type:
        """
        super().__init__()
        self.drl_name = 'swerc'
        self.gainer: SWERCGainManager = None
        self.rdwer: SWERCRDWManager = None
        self.resetter: SWERCResetter = None
        self.inputer = None
        self.__init_managers(input_type, reset_type)
        self.p_scene_size = np.array([])
        self.p_scene_diagonal_inv = 0
        self.v_scene_size = np.array([])
        self.v_scene_diagonal_inv = 0
        self.vel_min = 20 * TIME_STEP
        self.vel_max = 200 * TIME_STEP
        self.rot_mat = PI * TIME_STEP

        self.p_vis_poly: Polygon = None
        self.v_vis_poly: Polygon = None
        self.v_mapped_poly: Polygon = None
        self.pv_vis_poly: Polygon = None
        self.lidar_sample = lidar_num
        self.lidar_rot = np.linspace(-np.pi, np.pi, self.lidar_sample, endpoint=False)
        self.lidar_rot_atte = (2 * np.cos(self.lidar_rot) + 3) / 5  # shrunk to 0.2 - 1
        self.dir_energy_substrate = np.sum(self.lidar_rot_atte)

        self.lidar_ray_p_inters = []
        self.lidar_ray_v_inters = []
        self.lidar_ray_p_dis = []
        self.lidar_ray_v_dis = []
        self.lidar_ray_rela = []

        self.drl_agent: SWERCPPOAlg = SWERCPPOAlg()
        self.drl_agent.net_construct(net_width=512)
        self.drl_action = {
            'at': 1,
            'ar': 1,
            'ac': 1
            # 'srot': 0
        }

        self.drl_state = np.zeros(swerc_state_dim, dtype=np.float32)
        self.drl_stack_state = deque([np.zeros(self.drl_state.shape)] * swerc_state_seq, maxlen=swerc_state_seq)
        self.drl_pimg_state = np.zeros(swerc_img_state_dim, dtype=np.uint8)  # c,h,w
        self.drl_pimg_statck_state = deque([np.zeros(self.drl_pimg_state.shape)] * swerc_state_seq,
                                           maxlen=swerc_state_seq)
        self.drl_history = deque([], maxlen=30)
        self.reset_atte = 0
        self.step_atte = 0
        self.act_repeat_time = swerc_action_repeat
        self.act_update_counter = 0
        self.walked_dis = 0
        self.max_dis = swerc_max_v_dis

        self.epi_step = 0
        self.max_epi_step = self.drl_agent.train_config["max_epi_step"]
        self.acc_rwd = 0
        # --------------------------------image control---------------------------------------------------------
        self.g_inv = 1 / GRID_WIDTH
        self.hum_radius = int(HUMAN_RADIUS * self.g_inv)

        self.p_vis_center = np.array([MAX_PHY_WIDTH / 2, MAX_PHY_WIDTH / 2])
        self.p_space_surf: pygame.Surface = pygame.Surface((MAX_PHY_WIDTH, MAX_PHY_WIDTH), pygame.HWSURFACE)
        self.p_walkable_surf: pygame.Surface = pygame.Surface((MAX_PHY_WIDTH, MAX_PHY_WIDTH), pygame.HWSURFACE)
        self.p_cross_surf: pygame.Surface = pygame.Surface((MAX_PHY_WIDTH, MAX_PHY_WIDTH), pygame.HWSURFACE)
        self.p_loc_surf: pygame.Surface = pygame.Surface((MAX_PHY_WIDTH, MAX_PHY_WIDTH), pygame.HWSURFACE)
        self.p_trans_func = None

        self.v_vis_center = np.array([MAX_VIR_WIDTH / 2, MAX_VIR_WIDTH / 2])
        self.v_space_surf: pygame.Surface = pygame.Surface((MAX_VIR_WIDTH, MAX_VIR_WIDTH), pygame.HWSURFACE)
        self.v_walkable_surf: pygame.Surface = pygame.Surface((MAX_VIR_WIDTH, MAX_VIR_WIDTH), pygame.HWSURFACE)
        self.v_traj_surf: pygame.Surface = pygame.Surface((MAX_VIR_WIDTH, MAX_VIR_WIDTH), pygame.HWSURFACE)
        self.v_loc_surf: pygame.Surface = pygame.Surface((MAX_VIR_WIDTH, MAX_VIR_WIDTH), pygame.HWSURFACE)
        self.v_trans_func = None

    def set_manager(self, manager, m_type=None):
        """
        Prevent manual setting

        Args:
            manager:
            m_type:

        Returns:

        """
        pass

    def __init_managers(self, input_type, reset_type):
        self.gainer: SWERCGainManager = SWERCGainManager()
        self.rdwer: SWERCRDWManager = SWERCRDWManager()
        self.resetter: SWERCResetter = SWERCResetter(r_type=reset_type)
        self.inputer = SimuTrajectoryInputer()

        if input_type == 'traj':
            self.inputer = SimuTrajectoryInputer()
        elif input_type == 'live':
            self.inputer = LiveInputer()
        else:
            self.inputer = None
        assert self.inputer is not None, 'must assign an inputer: \'traj\' or \'live\''

        self.gainer.load_params()
        self.rdwer.load_params()
        self.resetter.load_params()
        self.inputer.load_params()

        self.gainer.setup_agent(self)
        self.rdwer.setup_agent(self)
        self.resetter.setup_agent(self)
        self.inputer.setup_agent(self)

    def prepare(self):
        super().prepare()
        c_rect = np.array(self.p_scene.bounds[0].cir_rect)
        v_rect = np.array(self.v_scene.bounds[0].cir_rect)
        self.p_scene_diagonal_inv = 1 / np.linalg.norm(np.max(c_rect, axis=0) - np.min(c_rect, axis=0))
        self.p_scene_size = np.array(self.p_scene.max_size)
        self.v_scene_diagonal_inv = 1 / np.linalg.norm(np.max(v_rect, axis=0) - np.min(v_rect, axis=0))
        self.v_scene_size = np.array(self.v_scene.max_size)

        self.step_atte = -self.p_scene.swe_weights.mean()
        self.reset_atte = self.step_atte * self.p_scene.swe_max_tr_dis / self.vel_max

        self.__prepare_pv_surfs()
        print('reset atte:', self.reset_atte)

    def reset(self):
        self.epi_step = 0
        self.walked_dis = 0
        self.acc_rwd = 0

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
        self.act_update_counter = 0

        self.drl_history.clear()
        self.__update_pv_vis_poly()
        self.step_state()
        self.drl_stack_state = deque([self.drl_state.copy()] * swerc_state_seq, maxlen=swerc_state_seq)

    def step_late(self, **kwargs):
        super().step_late()
        danger_dis = 200
        if self.resetter.reset_state:

            reward = self.reset_atte
        else:
            self.__update_pv_vis_poly()
            reward = 0
            rwd_loco = 0
            rwd_safe = 0

            if self.p_cur_vel > self.vel_min:
                rwd_loco, _ = self.p_scene.interpolate_tiling_weight_grad(self.p_cur_loc, self.p_scene.swe_weights)
                rwd_loco += self.step_atte

            if self.p_cur_rot > 0:
                for r_i in range(self.lidar_sample):
                    dir_att = self.lidar_rot_atte[r_i]
                    p_dis = self.lidar_ray_p_dis[r_i]
                    free_att = self.lidar_ray_rela[r_i]
                    safe_atte = np.log(danger_dis / p_dis) if p_dis >= 20 else 10
                    if safe_atte <= 0:
                        safe_atte = 0
                    rwd_safe += dir_att * pow(free_att, safe_atte)
                rwd_safe = (1 - rwd_safe / self.dir_energy_substrate) * self.step_atte

            reward += rwd_loco + rwd_safe

        self.acc_rwd += reward
        return reward

    def step_state(self):
        # vertical dim is meaningless for rdw
        self.walked_dis += self.v_cur_vel
        self.epi_step += 1
        self.drl_state = np.concatenate((self.p_cur_loc / self.p_scene_size,
                                         geo.norm_vec(self.p_cur_fwd),
                                         self.v_cur_loc / self.v_scene_size,
                                         geo.norm_vec(self.v_cur_fwd),
                                         np.array(self.lidar_ray_p_dis) * self.p_scene_diagonal_inv,
                                         np.array(self.lidar_ray_rela)), axis=None)
        self.drl_stack_state.append(self.drl_state)

    def state_refresh(self):
        """
        update all user states. Called after step_late.

        Returns:

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
            if not self.resetter.reset_state:
                s = np.array(self.drl_stack_state)
                if self.act_update_counter == 0:  # need update action
                    a, a_logprob = self.choose_action(s, True)
                    self.action_transition(*a, a_logprob)
                self.step_early()
                self.step(**self.drl_action)
                self.step_late()
                self.step_state()
                self.act_update_counter = (self.act_update_counter + 1) % self.act_repeat_time

                return (s,  # s_
                        done,
                        self.epi_step > self.max_epi_step,
                        None)
            else:  # current in reset state
                self.step_early()
                self.step()
                self.step_late()
                return None, False, False, None
        else:
            return None, done, False, None

    # -----------------------------------------------------------------------------------
    def action_transition(self, *args):
        self.drl_action = {
            'at': args[0],
            'ar': args[1],
            'ac': args[2],
            'log_prob': args[3]
        }

    def choose_action(self, state, deterministic=False):
        act, a_logprob = self.drl_agent.choose_action(state=state, deterministic=deterministic)
        return act, a_logprob

    def step_non_deterministic(self):
        self.p_cur_loc, self.p_cur_fwd, self.v_tar_loc, self.v_tar_fwd, done = self.inputer.update()
        # -------------------------------------
        # time of a frame, sync with inputer
        self.time_step = self.inputer.time_step
        self.gainer.time_step = self.time_step
        self.rdwer.time_step = self.time_step
        self.resetter.time_step = self.time_step
        # --------------------------------------
        if not done:
            if not self.resetter.reset_state:
                s = np.array(self.drl_stack_state)

                pop = False
                if self.act_update_counter == 0:  # need update action
                    a, a_logprob = self.choose_action(s)
                    self.action_transition(*a, a_logprob)
                    pop = True

                self.step_early()
                self.step(**self.drl_action)
                r = self.step_late()
                self.step_state()
                self.act_update_counter = (self.act_update_counter + 1) % self.act_repeat_time
                s_ = np.array(self.drl_stack_state)

                # notice that new action is also added to the history trajectory, so we focus on 0~n1 traj points
                self.drl_history.append({
                    "s": s,
                    "a": self.drl_action,
                    'r': r,
                    's_': s_,
                })
                truncated = self.epi_step > self.max_epi_step
                if (pop and len(self.drl_history) > 1) or truncated:
                    # the newest state is calculated after action update, so we need to push out this state and remain
                    # for the next traj head
                    cur_traj_head = self.drl_history.pop()
                    state_traj_start = self.drl_history[0]
                    state_traj_end = self.drl_history[-1]  # action has changed, only
                    s = state_traj_start['s']
                    a = (state_traj_start['a']['at'],
                         state_traj_start['a']['ar'],
                         state_traj_start['a']['ac'])
                    a_logprob = state_traj_start['a']['log_prob']
                    r = sum(map(lambda x: x['r'], self.drl_history)) / len(self.drl_history)
                    s_ = state_traj_end['s_']

                    self.drl_history.clear()
                    self.drl_history.append(cur_traj_head)

                    return (s,
                            a,
                            a_logprob,
                            r,
                            s_,
                            truncated,
                            truncated,
                            None)
            else:
                self.step_early()
                self.step()
                self.step_late()
            return (None,
                    None,
                    None,
                    0,
                    None,
                    False,
                    False,
                    None)
        else:
            return (None,
                    None,
                    None,
                    0,
                    None,
                    done,
                    False,
                    None)

    def __update_pv_vis_poly(self):
        # because v loc is updated after step(), need to recalculate cur v tiling
        self.v_cur_tiling, self.v_cur_conv = self.v_scene.calc_located_tiling_conv(self.v_cur_loc)
        self.p_vis_poly = self.p_cur_tiling.vis_poly[self.p_cur_tiling.obtain_vis_attr_id(self.p_cur_loc)]
        self.v_vis_poly = self.v_cur_tiling.vis_poly[self.v_cur_tiling.obtain_vis_attr_id(self.v_cur_loc)]
        trans_off = self.p_cur_loc - self.v_cur_loc
        self.v_mapped_poly = aff.translate(aff.rotate(geom=self.v_vis_poly,
                                                      angle=-geo.calc_angle_bet_vec(self.v_cur_fwd, self.p_cur_fwd),
                                                      origin=Point(self.v_cur_loc),
                                                      use_radians=True), trans_off[0], trans_off[1])

        inter_vis_poly = self.p_vis_poly.intersection(self.v_mapped_poly)
        if isinstance(inter_vis_poly, MultiPolygon):
            for geom in inter_vis_poly.geoms:
                if geom.contains(Point(self.p_cur_loc)):
                    self.pv_vis_poly = geom
        elif isinstance(inter_vis_poly, Polygon):
            self.pv_vis_poly = inter_vis_poly
        else:
            self.pv_vis_poly = None

        self.lidar_ray_p_inters = []
        self.lidar_ray_v_inters = []
        self.lidar_ray_p_dis = []
        self.lidar_ray_v_dis = []
        self.lidar_ray_rela = []

        for rot in self.lidar_rot:
            lidar_fwd = geo.rot_vecs(self.p_cur_fwd, rot)
            inter_p, p_dis = geo.calc_ray_poly_intersection(self.p_cur_loc, lidar_fwd, self.p_vis_poly)
            inter_v, v_dis = geo.calc_ray_poly_intersection(self.p_cur_loc, lidar_fwd, self.v_mapped_poly)
            self.lidar_ray_p_inters.append(inter_p)
            self.lidar_ray_v_inters.append(inter_v)
            self.lidar_ray_p_dis.append(p_dis)
            self.lidar_ray_v_dis.append(v_dis)
            self.lidar_ray_rela.append(alg.clamp(p_dis / v_dis, 0, 1) if v_dis != 0 else 1)

    # -----------------------------surfs images construct and rendering---------------------------------------------
    def __prepare_pv_surfs(self):
        max_p_size = int(max(self.p_scene.max_size) * self.g_inv + 2)  # 2 pixels more than the original size
        self.p_vis_center = np.array([max_p_size / 2, max_p_size / 2])
        self.p_space_surf: pygame.Surface = pygame.Surface((max_p_size, max_p_size), pygame.HWSURFACE)
        self.p_cross_surf: pygame.Surface = pygame.Surface((max_p_size, max_p_size), pygame.HWSURFACE)
        self.p_walkable_surf: pygame.Surface = pygame.Surface((max_p_size, max_p_size), pygame.HWSURFACE)
        self.p_loc_surf: pygame.Surface = pygame.Surface((max_p_size, max_p_size), pygame.HWSURFACE)
        self.p_loc_surf.fill(BLACK)
        self.p_cross_surf.fill(BLACK)
        self.p_walkable_surf.fill(BLACK)
        self.p_space_surf.fill(BLACK)

        self.p_trans_func = lambda x: ((np.array(x) - self.p_scene.scene_center) * self.g_inv +
                                       self.p_vis_center).astype(np.int32)



    def __update_pv_surfs(self):
        self.p_loc_surf.fill(BLACK)
        self.p_cross_surf.fill(BLACK)
        self.p_walkable_surf.fill(BLACK)

        p_vis_poly = self.p_trans_func(self.p_vis_poly.exterior.coords)
        pygame.gfxdraw.aapolygon(self.p_walkable_surf, p_vis_poly, WHITE)
        pygame.gfxdraw.filled_polygon(self.p_walkable_surf, p_vis_poly, WHITE)
        self.drl_pimg_state[0] = pygame.surfarray.array_red(pygame.transform.scale(self.p_walkable_surf, SHRUNK_SIZE))


    def render(self, wdn_obj: RDWWindow, default_color):
        super().render(wdn_obj, default_color)
        # steer_fwd = geo.rot_vecs(np.array([0, 1]), action_mapping(self.drl_action['srot'], -PI, PI)) * 100
        # wdn_obj.draw_phy_line(self.p_cur_loc, self.p_cur_loc + steer_fwd, 2, (0, 200, 0))

        # if self.p_vis_poly is not None and not self.p_vis_poly.is_empty and isinstance(self.p_vis_poly, Polygon):
        #     wdn_obj.draw_phy_poly(np.array(self.p_vis_poly.boundary.coords), fill=False, color=(125, 125, 125))
        #     wdn_obj.draw_phy_poly(np.array(self.v_mapped_poly.boundary.coords), fill=False, color=(200, 0, 0))
        # else:
        #     print(self.p_vis_poly)
        # if self.v_vis_poly is not None and not self.v_vis_poly.is_empty and isinstance(self.v_vis_poly, Polygon):
        #     wdn_obj.draw_vir_poly(np.array(self.v_vis_poly.boundary.coords), fill=False, color=(125, 125, 125))
        # if len(self.lidar_ray_p_inters) > 0:
        #     for i in range(len(self.lidar_ray_p_inters)):
        #         # inter_pv = self.lidar_ray_pv_inters[i]
        #         inter_p = self.lidar_ray_p_inters[i]
        #         inter_v = self.lidar_ray_v_inters[i]
        #         wdn_obj.draw_phy_line(self.p_cur_loc, inter_v, 2, color=(0, 255, 0))
        #         wdn_obj.draw_phy_line(self.p_cur_loc, inter_p, 2, color=(0, 255, 255))

        # -------------------------------directly transfer image to rdw window----------------------------------------
        # image_data = np.transpose(self.drl_pimg_state, axes=(2, 1, 0)).flatten()  #
        # p_img = pygame.transform.flip(pygame.image.frombuffer(image_data.tobytes(), SHRUNK_SIZE, 'RGBA'),
        #                               False,
        #                               True)
        # v_img = pygame.transform.flip(pygame.surfarray.make_surface(np.transpose(self.drl_vimg_state, axes=(1, 2, 0))),
        #                               False,
        #                               True)
        # wdn_obj.back_surf.blits([(p_img, (10, 10)),
        #                          (v_img, (400, 10))])
