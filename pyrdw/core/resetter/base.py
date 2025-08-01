import numpy as np

from pyrdw.core.resetter import *


class Turn21Resetter(BaseResetter):
    """
    优化后的Turn21重置，当其重置方向依然与边界碰撞时，将执行区域中心重置，四种方式中次优
    """

    def __init__(self):
        super().__init__()
        self.reset_type = 'Turn21'

    def calc_reset_target_fwd(self):
        self.reset_target_fwd = geo.rot_vecs(self.p_fwd, PI)


class TurnCenterResetter(BaseResetter):

    def __init__(self):
        super().__init__()
        self.reset_type = 'T2C'
        self.p_center = None

    def reset(self):
        super().reset()
        self.p_center = self.p_scene.bounds[0].center.copy()

    def calc_reset_target_fwd(self):
        self.reset_target_fwd = self.p_center - self.p_loc


class TurnModifiedCenterResetter(BaseResetter):
    """
    Proposed by https://ieeexplore.ieee.org/abstract/document/8797983
    """

    def __init__(self):
        super().__init__()
        self.reset_type = 'T2MC'
        self.p_center = None
        self.pot_rot_range= list(np.linspace(-PI,PI,360,endpoint=True))
        self.pot_rot_range1 = list(np.linspace(0, PI, 180, endpoint=True))
        self.pot_rot_range2 = list(np.linspace(0, -PI, 180, endpoint=True))

    def reset(self):
        super().reset()
        b_points = np.array(self.p_scene.bounds[0].points)
        min_xy = b_points.min(0)
        max_xy = b_points.max(0)
        self.p_center = (min_xy + max_xy) * 0.5

    def calc_reset_target_fwd(self):
        potential_fwd = self.p_center - self.p_loc
        if (LineString([self.p_loc, self.p_center]).intersects(self.p_scene.poly_contour.boundary)
                or alg.l2_norm(potential_fwd) == 0):
            potential_fwd = np.array([0,1])
            min_dis = float('inf')
            tar_fwd = None
            for rot in self.pot_rot_range:
                pot_fwd = geo.rot_vecs(potential_fwd,rot)
                pot_nx_ploc = self.p_loc+pot_fwd* self.reset_pred_t
                if (self.p_scene.poly_contour_safe.covers(Point(pot_nx_ploc))
                        and self.p_scene.poly_contour.covers(LineString([pot_nx_ploc,self.p_loc]))):
                        pot_dis =   alg.l2_norm(pot_nx_ploc-self.p_center)
                        if pot_dis<min_dis:
                            min_dis = pot_dis
                            tar_fwd = pot_fwd
            if tar_fwd is not None:
                self.reset_target_fwd = tar_fwd
            else:
                self.reset_target_fwd = geo.rot_vecs(self.p_fwd, PI)

            # inter, dis, bound = geo.calc_ray_poly_intersection_bound(self.p_loc, potential_fwd, self.p_scene.bounds)
            # bs, be = bound
            # potential_f1 = geo.norm_vec(be - bs)
            # potential_f2 = -potential_f1
            # pot_end1 = self.p_loc + potential_f1
            # pot_end2 = self.p_loc + potential_f2
            # if alg.l2_norm(pot_end2 - self.p_center) > alg.l2_norm(pot_end1 - self.p_center):
            #     self.reset_target_fwd = potential_f1
            # else:
            #     self.reset_target_fwd = potential_f2
            #
            # cur_reset_fwd = geo.norm_vec(self.reset_target_fwd)
            # nx_pot_loc = self.p_loc + cur_reset_fwd * self.reset_pred_t
            #
            # pot_rot_range1 = copy.copy(self.pot_rot_range1)
            # pot_rot_range2 = copy.copy(self.pot_rot_range2)
            #
            # while not self.p_scene.poly_contour_safe.covers(Point(nx_pot_loc)):
            #     pot_rot1 = pot_rot_range1.pop()
            #     pot_rot2 = pot_rot_range2.pop()
            #     pot_reset_fwd1 = geo.rot_vecs(cur_reset_fwd, pot_rot1)
            #     pot_reset_fwd2 = geo.rot_vecs(cur_reset_fwd, pot_rot2)
            #     pot_reset_end1 = self.p_loc + pot_reset_fwd1
            #     pot_reset_end2 = self.p_loc + pot_reset_fwd2
            #     if alg.l2_norm(pot_reset_end2 - self.p_center) > alg.l2_norm(pot_reset_end1 - self.p_center):
            #         cur_reset_fwd = pot_reset_fwd1
            #     else:
            #         cur_reset_fwd = pot_reset_fwd2
            #     nx_pot_loc = self.p_loc + cur_reset_fwd * self.reset_pred_t
            #
            #
            #     self.reset_target_fwd = cur_reset_fwd
        else:
            self.reset_target_fwd = potential_fwd


class TurnSteerForceResetter(BaseResetter):

    def __init__(self):
        super().__init__()
        self.reset_type = 'T2SF'
        self.p_center = None

    def reset(self):
        super().reset()
        self.p_center = self.p_scene.bounds[0].center.copy()

    def calc_reset_target_fwd(self):
        self.reset_target_fwd = np.array(self.agent.rdwer.steer_force).copy()
        if self.reset_target_fwd[0] == 0 and self.reset_target_fwd[1] == 0:
            self.reset_target_fwd = self.p_center - self.p_loc


class TurnAPFGradientResetter(TurnSteerForceResetter):

    def __init__(self):
        super().__init__()
        self.reset_type = 'T2APFG'
        self.p_center = None
        self.eta = 1
        self.repulsion_range = 10  # m

    def calc_pos_apf_grad(self, pos, near_obs):
        p2obs = pos - near_obs
        dis2obs = alg.l2_norm(p2obs)
        repulsion_f = 100 / dis2obs  # convert to meter
        repulsion_f_grad = p2obs * repulsion_f
        return self.eta * (repulsion_f - 1 / self.repulsion_range) * (repulsion_f ** 2) * repulsion_f_grad

    def calc_reset_target_fwd(self):
        self.reset_target_fwd = self.calc_pos_apf_grad(self.p_loc, self.agent.p_cur_tiling.nearst_obs_pos)


class TurnAPFGradientStepForwardResetter(TurnAPFGradientResetter):

    def __init__(self):
        super().__init__()
        self.reset_type = 'T2APFG-SF'
        self.p_center = None
        self.eta = 1
        self.repulsion_range = 10
        self.step_num = 2
        self.step_size = 0.2

    def calc_reset_target_fwd(self):
        super().calc_reset_target_fwd()
        inten_loc = self.p_loc + geo.norm_vec(self.reset_target_fwd) * self.step_num * self.step_size
        if not self.p_scene.poly_contour.intersects(LineString([self.p_loc, inten_loc])):
            cur_grad = self.reset_target_fwd
            cur_loc = self.p_loc
            delta_step = self.step_size * TIME_STEP
            total_steps = int(self.step_num * (1 / TIME_STEP))

            for _ in range(total_steps):
                next_loc = cur_loc + geo.norm_vec(cur_grad) * delta_step
                next_grad = self.calc_pos_apf_grad(next_loc, self.agent.p_cur_tiling.nearst_obs_pos)
                cur_loc = next_loc
                cur_grad = next_grad
            self.reset_target_fwd = cur_loc - self.p_loc  # cur_grad

