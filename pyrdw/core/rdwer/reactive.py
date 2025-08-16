import numpy as np

import pyrdw.core.gain.base
from pyrdw.core.rdwer import *



class NoRdwManager(BaseRdwManager):

    def __init__(self):
        super(NoRdwManager, self).__init__()
        self.rdw_type = 'No Rdw'
        # Reactive redirection velocity constants, index 0 is virtual view auto-rotation speed when static, index 1 is static-move distinction threshold, index 2 is gc maximum angular velocity change, index 3 is gr maximum angular velocity change
        self.steer_static_rot = 4  # Static scene auto-rotation angle of classic Steer RDW algorithm
        self.steer_vel_dt = 20  # Walking detection threshold of classic Steer RDW algorithm, default 0.2m/s to distinguish whether user is walking or static
        self.steer_rot_dt = 1.5  # in degree
        self.steer_max_gc_rot = 15  # Upper limit of curvature gain rotation angle that classic Steer RDW algorithm can apply
        self.steer_max_gr_rot = 30  # Upper limit of rotation gain rotation angle that classic Steer RDW algorithm can apply
        self.steer_dampen_dis = 125  # Reactive redirection damping term, distance damping term
        self.steer_dampen_ang = 45  # Reactive redirection damping term, angle damping term
        self.steer_force = np.array([0, 0])
        self.enable_rdw = False

    def load_params(self):
        self.steer_static_rot = self.rdw_spec['static_rot'] * DEG2RAD
        self.steer_vel_dt = self.rdw_spec['move_dt']
        self.steer_rot_dt = self.rdw_spec['rot_dt'] * DEG2RAD
        self.steer_max_gc_rot = self.rdw_spec['max_gc_rot'] * DEG2RAD
        self.steer_max_gr_rot = self.rdw_spec['max_gr_rot'] * DEG2RAD
        self.steer_dampen_dis = self.rdw_spec['dampen_dis']
        self.steer_dampen_ang = self.rdw_spec['dampen_bear'] * DEG2RAD

    def reset(self):
        self.enable_rdw = False

    def update(self, **kwargs):
        pass

    def calc_vir_state(self):
        v_mov_vec = self.p_vel_vec
        vir_next_fwd = geo.norm_vec(geo.rot_vecs(self.v_fwd, self.p_rot))
        self.agent.v_cur_loc = self.v_loc + v_mov_vec
        self.agent.v_cur_fwd = vir_next_fwd
        self.agent.v_cur_vel = self.p_vel
        self.agent.v_cur_rot = self.p_rot

    def render(self, wdn_obj, default_color):
        pass


class SteerRdwManager(NoRdwManager):

    def __init__(self):
        super().__init__()
        # self.gc = 1 / 750  # default curvature gain
        # self.gr = [0.85, 1.3]
        self.last_inject_rot = 0
        self.use_bear2target = True
        self.use_dis2target = True

    def reset(self):
        super().reset()
        self.enable_rdw = True
        self.last_inject_rot = 0

    def calc_vir_state(self):
        gain_mg = self.agent.gainer
        p_fwd_vec, p_fwd_vel, p_ver_vec, p_ver_vel = self.p_vel_decompose()

        if self.enable_rdw:
            desired_rotation = geo.calc_angle_bet_vec(self.p_fwd, self.steer_force)
            if desired_rotation != 0:
                desired_rot_dir = desired_rotation / abs(desired_rotation)  # + 顺时针 - 逆时针
            else:
                desired_rot_dir = 0
            # you need to steer users to opposite dir in vir scene, so that they will steer correctly in phy scene
            desired_rot_dir *= -1
            m_rot_inject = 0

            if p_fwd_vel > self.steer_vel_dt * self.time_step:
                # 基于行走速度的方向偏转
                m_rot_inject = alg.clamp(abs(p_fwd_vel * gain_mg.gc_const[0][1] * 0.01), 0,
                                         self.steer_max_gc_rot * self.time_step)
            h_rot_inject = 0
            if abs(self.p_rot) > self.steer_rot_dt * self.time_step:
                if self.p_rot * desired_rot_dir > 0:  # rotation direction is with the desired direction
                    applied_gr = gain_mg.gr_const[0][1]
                else:
                    applied_gr = gain_mg.gr_const[0][0]
                h_rot_inject = alg.clamp(abs(self.p_rot * (applied_gr - 1)), 0, self.steer_max_gr_rot * self.time_step)

            rot_inject = desired_rot_dir * max(m_rot_inject, h_rot_inject)

            scale_factor = 1
            bearing_to_target = abs(desired_rotation)
            length_to_target = alg.l2_norm(self.steer_force)

            if bearing_to_target < self.steer_dampen_ang and self.use_bear2target:
                scale_factor *= math.sin(PI_1_2 * bearing_to_target / self.steer_dampen_ang)

            if length_to_target < self.steer_dampen_dis and self.use_dis2target:
                scale_factor *= length_to_target / self.steer_dampen_dis

            rot_inject *= scale_factor

            vir_rot_vel = self.p_rot + rot_inject
            vir_fwd_vel = p_fwd_vel
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
        wdn_obj.draw_phy_line(self.p_loc, self.steer_force + self.p_loc, 2, (255, 0, 0))

    def copy_target_manager(self, other_mg):
        if other_mg is not None:
            self.steer_static_rot = other_mg.steer_static_rot
            self.steer_vel_dt = other_mg.steer_vel_dt
            self.steer_rot_dt = other_mg.steer_rot_dt
            self.steer_max_gc_rot = other_mg.steer_max_gc_rot
            self.steer_max_gr_rot = other_mg.steer_max_gr_rot
            self.steer_dampen_dis = other_mg.steer_dampen_dis
            self.steer_dampen_ang = other_mg.steer_dampen_ang
        else:
            self.load_params()


class S2CRdwManager(SteerRdwManager):
    """
    参考https://ieeexplore.ieee.org/abstract/document/6479192

    """

    def __init__(self):
        super().__init__()
        self.rdw_type = 'S2C Rdw'

    def update(self, **kwargs):
        self.steer_force = np.array(self.p_scene.max_size) / 2 - self.p_loc


class S2ORdwManager(SteerRdwManager):
    """
    参考https://ieeexplore.ieee.org/abstract/document/6479192


    """

    def __init__(self):
        super().__init__()
        self.rdw_type = 'S2O Rdw'
        self.steer_circle = [[0, 0], 10]

    def update(self, **kwargs):
        self.steer_circle = self.agent.p_cur_conv.in_circle
        r = self.steer_circle[1] * 0.8
        center = np.array(self.steer_circle[0])
        p2center = center - self.p_loc
        m, n = p2center
        dis2center = alg.l2_norm(p2center)
        if dis2center > r:
            a = dis2center ** 2
            b = 2 * r * n
            c = r * r - m * m
            sin_theta1 = (-b + (b * b - 4 * a * c) ** 0.5) / (2 * a)
            sin_theta2 = (-b - (b * b - 4 * a * c) ** 0.5) / (2 * a)
            cx, cy = self.steer_circle[0]
            y1 = sin_theta1 * r + cy
            y2 = sin_theta2 * r + cy
            x1 = (1 - sin_theta1 ** 2) ** 0.5 * r + cx
            x2 = -(1 - sin_theta1 ** 2) ** 0.5 * r + cx
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y1])
            vt1 = self.p_loc - p1
            vt2 = center - p1
            vt3 = self.p_loc - p2
            vt4 = center - p2
            cos_t1 = abs(vt1 @ vt2 / (alg.l2_norm(vt1) * alg.l2_norm(vt2)))
            cos_t2 = abs(vt3 @ vt4 / (alg.l2_norm(vt3) * alg.l2_norm(vt4)))
            if cos_t1 < cos_t2:
                tar1 = np.array([x1, y1])
            else:
                tar1 = np.array([x2, y1])
            x1 = (1 - sin_theta2 ** 2) ** 0.5 * r + cx
            x2 = -(1 - sin_theta2 ** 2) ** 0.5 * r + cx
            p1 = np.array([x1, y2])
            p2 = np.array([x2, y2])
            vt1 = self.p_loc - p1
            vt2 = center - p1
            vt3 = self.p_loc - p2
            vt4 = center - p2
            cos_t1 = abs(vt1 @ vt2 / (alg.l2_norm(vt1) * alg.l2_norm(vt2)))
            cos_t2 = abs(vt3 @ vt4 / (alg.l2_norm(vt3) * alg.l2_norm(vt4)))
            if cos_t1 < cos_t2:
                tar2 = np.array([x1, y2])
            else:
                tar2 = np.array([x2, y2])
        else:  # 当前点在圆内，则求以该点与圆心连线为中心线，与中心线左右两侧夹角为60度的两条直线和圆的交点,所得点之前圆弧为优弧
            a = 4
            b = 6 * dis2center
            c = 3 * dis2center ** 2 - r ** 2
            ml = (-b + (b * b - 16 * c) ** 0.5) / 8
            nl = (r * r - ml * ml) ** 0.5

            v = geo.norm_vec(p2center)
            pm = center + v * ml
            v_nl = v * nl
            tar1 = np.array([pm[0] - v_nl[1], pm[1] + v_nl[0]])
            tar2 = np.array([pm[0] + v_nl[1], pm[1] - v_nl[0]])
        ap1 = abs(geo.calc_angle_bet_vec(tar1 - self.p_loc, self.p_fwd))
        ap2 = abs(geo.calc_angle_bet_vec(tar2 - self.p_loc, self.p_fwd))
        tar = tar1 if ap1 < ap2 else tar2
        self.steer_force = tar - self.p_loc

    def render(self, wdn_obj, default_color):
        wdn_obj.draw_phy_line(self.p_loc, self.steer_force + self.p_loc, 2, (255, 0, 0))
        r = self.steer_circle[1] * 0.8
        center = np.array(self.steer_circle[0])
        wdn_obj.draw_phy_circle(center, r, (255, 0, 0))


class APFRdwManager(SteerRdwManager):
    """
    计算某点人工势场负梯度
    https://zhuanlan.zhihu.com/p/434095158
    https://blog.csdn.net/qq_44339029/article/details/128510395
    https://ieeexplore.ieee.org/document/8797983
    """

    def __init__(self):
        super(APFRdwManager, self).__init__()
        self.rdw_type = 'APF Rdw'
        self.apf_force_c = np.array([1, 1])  # 人工势场斥力项和引力项各自系数 [repulsion_coefficient,gravitation_coefficient]
        self.eta = 1
        self.repulsion_range = 10  # m

        # self.steer_force = [0, 0]
        # self.apf_reps_set = []  # apf斥力集合
        # self.apf_gras_set = []  # apf引力集合
        # self.apf_forces = [[0, 0], [0, 0]]  # 人工势场斥力合力和引力合力

    def load_params(self):
        super().load_params()
        self.apf_force_c = [self.rdw_spec['apf_rep_ft'], self.rdw_spec['apf_gra_ft']]

    def update(self, **kwargs):
        p2obs = (self.p_loc - self.agent.p_cur_tiling.nearst_obs_pos) * 0.01
        dis2obs = alg.l2_norm(p2obs)
        repulsion_f = 1 / dis2obs
        repulsion_f_grad = p2obs * repulsion_f
        repulsion_grad = self.eta * (repulsion_f - 1 / self.repulsion_range) * (repulsion_f ** 2) * repulsion_f_grad
        self.steer_force = geo.norm_vec(repulsion_grad) * 100

    def calc_vir_state(self):
        gain_mg = self.agent.gainer
        p_fwd_vec, p_fwd_vel, p_ver_vec, p_ver_vel = self.p_vel_decompose()

        if self.enable_rdw:
            if p_fwd_vel > self.steer_vel_dt * self.time_step:  # 当前正在行走
                gt = gain_mg.gt_const[0][0] if self.steer_force @ self.p_fwd < 0 else gain_mg.gt_const[0][1]
                gc = gain_mg.gc_const[0][0] if geo.calc_angle_bet_vec(self.p_fwd, self.steer_force) > 0 else \
                    gain_mg.gc_const[0][1]  # need to push the user towards steer force
                gr = 0
            else:  # 当前正在原地旋转
                gt = 1
                gc = 0
                cur_ang = geo.calc_angle_bet_vec(self.p_fwd, self.steer_force)
                lst_ang = geo.calc_angle_bet_vec(self.agent.p_lst_fwd, self.steer_force)
                if cur_ang * lst_ang > 0:
                    if abs(cur_ang) < abs(lst_ang):
                        gr = gain_mg.gr_const[0][1]
                    else:
                        gr = gain_mg.gr_const[0][0]
                else:
                    gr = gain_mg.gr_const[0][1]

            vir_rot_vel = self.p_rot * gr + p_fwd_vel * gc * 0.01  # need to scale p_vel from cm to m
            vir_fwd_vel = p_fwd_vel * gt
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
        wdn_obj.draw_phy_line(self.p_loc, self.steer_force + self.p_loc, 2, (255, 0, 0))
        wdn_obj.draw_phy_circle(self.agent.p_cur_tiling.nearst_obs_pos, 10, (255, 0, 0))

    def copy_target_manager(self, other_mg):
        if other_mg is not None:
            self.steer_static_rot = other_mg.steer_static_rot
            self.steer_vel_dt = other_mg.steer_vel_dt
            self.steer_rot_dt = other_mg.steer_rot_dt
            self.steer_max_gc_rot = other_mg.steer_max_gc_rot
            self.steer_max_gr_rot = other_mg.steer_max_gr_rot
            self.steer_dampen_dis = other_mg.steer_dampen_dis
            self.steer_dampen_ang = other_mg.steer_dampen_ang
            self.apf_force_c = pickle.loads(pickle.dumps(other_mg.apf_force_c))
        else:
            self.load_params()





