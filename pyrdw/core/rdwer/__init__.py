from pyrdw.core import *


class BaseRdwManager(BaseManager):

    def __init__(self):
        super().__init__()
        self.rdw_type = None
        self.rdw_spec = const_steer
        self.enable_rdw = True

    def load_params(self):
        pass

    def prepare(self):
        self.p_scene, self.v_scene = self.agent.p_scene, self.agent.v_scene

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calc_vir_state(self):

        raise NotImplementedError

    def p_vel_decompose(self):
        """
        Decompose phy vel vector v_vel, one part is following current phy head forward (self.p_fwd) v_fwd,
        other v_ver is vertical to v_fwd.

        实现物理空间速度向量的正交分解，用于分析物体前向和横向运动

        Returns:
            - p_fwd_vec (normalized),
            - p_fwd_vel,
            p_right_vec (normalized),
            p_right_vel
        """
        p_fwd_vec = geo.norm_vec(self.p_fwd)
        p_fwd_vel = np.dot(p_fwd_vec, self.p_vel_vec)

        p_ver_vec = self.p_vel_vec - p_fwd_vel * p_fwd_vec
        p_ver_vel = alg.l2_norm(p_ver_vec)
        p_ver_vec = geo.norm_vec(p_ver_vec)
        return p_fwd_vec, p_fwd_vel, p_ver_vec, p_ver_vel

    def copy_target_manager(self, other_mg):
        pass
