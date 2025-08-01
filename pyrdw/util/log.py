from pyrdw.core import *
from pyrdw.core.agent.base import BaseAgent


class Logger(ABC):

    def __init__(self, agent):
        self.agent: BaseAgent = agent
        self.delta_time = const_env['time_step']
        self.rdw_type = 'none'
        self.reset_type = 'none'
        self.reset_state = 'normal'
        self.v_scene_size = None
        self.p_scene_size = None
        self.p_walk_dis = 0
        self.v_walk_dis = 0
        self.p_cur_vel = np.array([0, 0])
        self.v_cur_vel = np.array([0, 0])
        self.rdw_total_rate = np.array([0, 0])
        self.g_rot_range = [0, 0]
        self.g_tran_range = [0, 0]
        self.g_curv_range = [0, 0]
        self.gains = [0, 0, 0, 0]
        self.gains_rate = [0, 0, 0, 0]
        self.p_pos_record = []
        self.reset_pos_record = []
        self.past_reset_num = []
        self.reset_num = 0
        self.max_reset_num = 0
        self.total_reset_num = 0
        self.cur_step = 0
        self.episode_num = 0
        self.max_record_len = 100
        self.past_epi_frames = None

    def set_agent(self, agent: BaseAgent):
        self.agent = agent

    def set_epi_data_buffer(self, max_rec_len):
        self.max_record_len = max_rec_len
        self.past_reset_num = [0] * self.max_record_len
        self.past_epi_frames = [0] * self.max_record_len

    @abstractmethod
    def prepare_log(self):
        raise NotImplementedError

    @abstractmethod
    def reset_epi_data(self):
        raise NotImplementedError

    @abstractmethod
    def record_step_data(self):
        raise NotImplementedError

    @abstractmethod
    def log_epi_data(self):
        raise NotImplementedError


class RDWBaseLogger(Logger):

    def __init__(self, agent):
        super().__init__(agent)
        self.epi_states = []

    def prepare_log(self):
        """
        clean up total reset information in prepare.

        Returns:

        """
        self.total_reset_num = 0

    def reset_epi_data(self):
        self.epi_states = []
        self.rdw_type = self.agent.rdwer.rdw_type
        self.v_scene_size = self.agent.v_scene.max_size[:]
        self.p_scene_size = self.agent.p_scene.max_size[:]
        self.cur_step = 0
        self.p_walk_dis = 0
        self.v_walk_dis = 0
        self.p_cur_vel = np.array([0, 0])
        self.v_cur_vel = np.array([0, 0])
        self.rdw_total_rate = []
        self.g_rot_range = self.agent.gainer.gr_range[0].tolist()
        self.g_tran_range = self.agent.gainer.gt_range[0].tolist()
        self.g_curv_range = self.agent.gainer.gc_range[0].tolist()
        self.p_pos_record = []
        self.reset_pos_record = []
        self.reset_num = 0

    def record_step_data(self):
        step_state = self.agent.state_log()
        self.epi_states.append(step_state)
        self.cur_step = step_state['t']
        if not step_state['rs_s']:
            self.rdw_total_rate.append(step_state['rdw_r'])
            self.p_walk_dis += step_state['pwv']
            self.v_walk_dis += step_state['vwv']

    def log_epi_data(self):
        self.reset_num = self.epi_states[-1]['rs_n']
        self.total_reset_num += self.reset_num
        states = []
        for s in self.epi_states:
            new_s = {
                't': s['t'],  # current step
                'pwv': s['pwv'],  # p walking velocity
                'prv': s['prv'],  # p rotating velocity
                'p_loc': np.round(s['p_loc'], decimals=1).tolist(),  # p location
                'p_fwd': np.round(s['p_fwd'], decimals=1).tolist(),  # p forward
                'v_loc': np.round(s['v_loc'], decimals=1).tolist(),
                'v_fwd': np.round(s['v_fwd'], decimals=1).tolist(),
                'vwv': s['vwv'],
                'vrv': s['vrv'],
                'rdw_r': s['rdw_r'],
                'rs_s': s['rs_s'],
                'rs_t': s['rs_t'],
                'rs_n': s['rs_n'],
                'rs_info': s['rs_info'],
            }
            states.append(new_s)

        data = {
            'v_name': self.agent.v_scene.name,
            'p_name': self.agent.p_scene.name,
            'traj_type': self.agent.inputer.cur_traj.type if self.agent.inputer.cur_traj is not None else 'live',
            'alg_name': self.agent.name,
            'reset_num': self.reset_num,
            'total_reset_num': self.total_reset_num,  # reset number since prepare function
            'mean_rdw_rate': np.array(self.rdw_total_rate).mean() / self.delta_time * RAD2DEG,
            'walk_dis': self.p_walk_dis * 0.01,
            'avg_dis_btw_resets': self.p_walk_dis * 0.01 / (self.reset_num + 1),
            'v_walk_dis': self.v_walk_dis * 0.01,
            'vdis_btw_resets': self.v_walk_dis * 0.01 / (self.reset_num + 1),
            'state_traj': states  # 该回合完整的状态轨迹
        }
        return data


class RDWTrajectoryWalkerLogger(RDWBaseLogger):

    def __init__(self, agent):
        super().__init__(agent)
        self.cur_traj_type = None
        self.total_traj_num = 0
        self.cur_traj_idx = 0
        self.total_tar_num = 0
        self.cur_tar_idx = 0

    def record_step_data(self):
        super().record_step_data()

#
#
# class RDWDRLLogger(Logger):
#
#     def __init__(self, env):
#         super(RDWDRLLogger, self).__init__(env)
#         self.delta_time = 0
#         self.rdw_env_mg = None
#         self.rdw_env = None
#         self.rdw_type = None
#         self.reset_state = 'normal'
#         self.reset_type = 'none'
#         self.v_scene_size = None
#         self.p_scene_size = None
#         self.phy_cur_vel = np.array([0, 0])
#         self.vir_cur_vel = np.array([0, 0])
#         self.rdw_cur_rate = np.array([0, 0])
#         self.rdw_total_rate = np.array([0, 0])
#         self.rdw_mean_rate = np.array([0, 0])
#         self.gr_range = [0, 0]
#         self.gt_range = [0, 0]
#         self.gc_range = [0, 0]
#         self.gains = [0, 0, 0, 0]
#         self.gains_rate = [0, 0, 0, 0]
#         self.past_reset_num = []
#         self.reset_num = 0
#         self.max_reset_num = 0
#         self.total_reset_num = 0
#         self.cur_traj_type = None
#         self.total_traj_num = 0
#         self.cur_traj_idx = 0
#         self.total_tar_num = 0
#         self.cur_tar_idx = 0
#
#     def set_agent(self, env_mg):
#         self.rdw_env_mg = env_mg
#         self.rdw_env = self.rdw_env_mg.env
#
#     def set_epi_data_buffer(self, max_rec_len):
#         self.max_record_len = max_rec_len
#         self.past_epi_frames = [0] * self.max_record_len
#         self.past_mean_ls = [0] * self.max_record_len
#         self.past_total_rwd = [0] * self.max_record_len
#         self.past_reset_num = [0] * self.max_record_len
#
#     def reset_epi_data(self, env):
#         self.rdw_env = self.rdw_env_mg.env
#         self.rdw_type = self.rdw_env.rdwer.rdw_type
#         self.v_scene_size = self.rdw_env.v_scene.max_size[:]
#         self.p_scene_size = self.rdw_env.p_scene.max_size[:]
#         self.cur_mean_ls = 0
#         self.cur_ls = 0
#         self.cur_rwd = 0
#         self.cur_total_reward = 0
#         self.cur_step = 0
#         self.phy_cur_vel = np.array([0, 0])
#         self.vir_cur_vel = np.array([0, 0])
#         self.rdw_cur_rate = np.array([0, 0])
#         self.rdw_total_rate = np.array([0, 0])
#         self.rdw_mean_rate = np.array([0, 0])
#         self.gr_range = self.rdw_env.gainer.gr_range[0].tolist()
#         self.gt_range = self.rdw_env.gainer.gt_range[0].tolist()
#         self.gc_range = self.rdw_env.gainer.gc_range[0].tolist()
#         self.reset_num = 0
#         self.cur_traj_type = self.rdw_env_mg.trajs_cur_type
#         self.total_traj_num = len(self.rdw_env.inputer.road_targets_lists)
#         self.cur_traj_idx = 0
#         self.total_tar_num = len(self.rdw_env.inputer.cur_tars)
#         self.cur_tar_idx = 0
#         self.delta_time = 1 / TIME_STEP
#
#     def record_epi_data(self, ls, rwd, done):
#         self.cur_step += 1
#         # --------------记录强化学习信息-----------------------------------
#         if ls is not None:
#             self.cur_mean_ls += ls
#             self.cur_ls = ls
#         self.cur_rwd = rwd
#         self.cur_total_reward += rwd
#         if self.cur_rwd > self.max_single_rwd:
#             self.max_single_rwd = self.cur_rwd
#         # ------------------------------记录重定向信息------------------------------------------------------------------
#         if self.rdw_env.walker_type == WalkerType.TrajectoryWalker.value:
#             self.cur_traj_idx = self.rdw_env_mg.cur_traj_idx
#             self.cur_tar_idx = self.rdw_env.inputer.cur_tar_idx
#         self.phy_cur_vel = np.array([self.rdw_env.p_cur_vel, self.rdw_env.p_cur_rot]) * self.delta_time
#         self.vir_cur_vel = np.array([self.rdw_env.v_cur_vel, self.rdw_env.v_cur_rot]) * self.delta_time
#         self.gains = self.rdw_env.gainer.g_values
#         self.gains_rate = self.rdw_env.gainer.g_rates
#         if not self.rdw_env.resetter.reset_state:
#             self.reset_state = 'normal'
#             self.reset_type = 'none'
#             self.rdw_cur_rate[0] = self.vir_cur_vel[0] - self.phy_cur_vel[0]  # 当前速度变化率
#             self.rdw_cur_rate[1] = abs(self.vir_cur_vel[1]) - abs(self.phy_cur_vel[1])  # 当前旋转变化率（重定向率）
#             self.rdw_total_rate += self.rdw_cur_rate
#             self.rdw_mean_rate = self.rdw_total_rate / self.cur_step
#         else:
#             self.reset_state = 'reset'
#             self.reset_type = self.rdw_env.resetter.reset_type
#             # print(self.rdw_env.resetter.reset_pre_state)
#             if not self.rdw_env.resetter.reset_pre_state:
#                 self.reset_num += 1
#
#     def log_epi_data(self):
#         self.cur_mean_ls = self.cur_mean_ls / self.cur_step
#         self.episode_num += 1
#         epi_num = self.episode_num % self.max_record_len
#         self.past_epi_frames[epi_num] = self.cur_step
#         self.past_mean_ls[epi_num] = self.cur_mean_ls
#         self.past_total_rwd[epi_num] = self.cur_total_reward
#         self.past_reset_num[epi_num] = self.reset_num
#         if self.cur_total_reward > self.max_total_rwd:
#             self.max_total_rwd = self.cur_total_reward
#         if self.reset_num > self.max_reset_num:
#             self.max_reset_num = self.reset_num
#         self.total_reset_num += self.reset_num
#         if self.writer is not None:
#             self.writer.add_scalars('evl', {'total_rwd': self.cur_total_reward, 'mean_ls': self.cur_mean_ls},
#                                     self.episode_num)
#         d = datetime.datetime.now()
#         print("{}:{}:{}  ".format(d.hour, d.minute, d.second), end="")
#         if self.training_mark:
#             print('train_epi:{}, frames:{}, reward:{:.4f}, loss:{:.4f}'.format(
#                 self.episode_num,
#                 self.past_epi_frames[epi_num],
#                 self.past_total_rwd[epi_num],
#                 self.past_mean_ls[epi_num]))
#         else:
#             print('eval_epi, frames:{}, reward:{}'.format(self.past_epi_frames[epi_num], self.past_total_rwd[epi_num]))
#         print('\t\trdw info- resets:{}, vel rdw rate:{:.4f}, rot rdw rate:{:.4f}'.format(self.reset_num,
#                                                                                          *self.rdw_mean_rate))
#
#     def log_vir_bounds(self):
#         print("max width=", self.rdw_env.v_scene.max_size[0], "max height=", self.rdw_env.v_scene.max_size[1])
#         for bound in self.rdw_env.v_scene.bounds:
#             print("*****boundary*****")
#             bound.print_info()
#             print("\n")
#
#     def log_vir_patches(self):
#         for patch in self.rdw_env.v_scene.patches:
#             str_info = "patch id:" + str(patch.id) + "\n"
#             str_info += "  start: " + str(patch.start_node.id) + " " + str(patch.start_node.pos) + "\n"
#             if patch.start_node.rela_loop_node is not None:
#                 str_info += "    loop connect: " + str(patch.start_node.rela_loop_node.id) + "\n"
#             str_info += "  end: " + str(patch.end_node.id) + " " + str(patch.end_node.pos) + "\n"
#             if patch.end_node.rela_loop_node is not None:
#                 str_info += "    loop connect: " + str(patch.end_node.rela_loop_node.id) + "\n"
#             str_info += "  all node: "
#             for node in patch.nodes:
#                 str_info = str_info + " " + str(node.id)
#             str_info += "\n  start neighbor: "
#             for i in range(len(patch.start_nb_patches)):
#                 other_patch = patch.start_nb_patches[i]
#                 str_info += str(other_patch.id) + " " + str(patch.start_nb_patches_order[i]) + " "
#             str_info += "\n  end neighbor:"
#             for i in range(len(patch.end_nb_patches)):
#                 other_patch = patch.end_nb_patches[i]
#                 str_info += str(other_patch.id) + " " + str(patch.end_nb_patches_order[i]) + " "
#             print(str_info)
#
#     def log_vir_tilings(self):
#         print("tiling num:", self.rdw_env.v_scene.tilings_shape[0] * self.rdw_env.v_scene.tilings_shape[1])
#         for tiling in self.rdw_env.v_scene.tilings:
#             tiling_str = ""
#             id = tiling.id
#             for n_tiling in tiling.neighbors:
#                 n_id = n_tiling.tiling_id
#                 tiling_str = tiling_str + " " + str(n_id)
#             print("tiling id:", id, "neighbors ids:", tiling_str)
#
#     def log_env_infos(self):
#         print("time step - {} s".format(TIME_STEP))
#
#     def log_scene_infos(self, s_type="vir"):
#         scene = self.rdw_env.v_scene if s_type == "vir" else self.rdw_env.p_scene
#         print("*-------------------------{} scene info-----------------------------".format('vir'))
#         print("* Scene name: {}".format(scene.name))
#
#     def log_rdw_gains(self):
#         mg = self.rdw_env.gainer
#
#         print("*-------------------------Rdw gains info-----------------------------")
#         print("*  gain controller type - " + str(type(self.rdw_env.gainer).__name__))
#         print("*Translation Gain:")
#         print("*  tight  -current [{}, {}]  -default [{}, {}]".format(mg.gt_const[0][0], mg.gt_const[0][1],
#                                                                       mg.gt_const[0][2], mg.gt_const[0][3]))
#         print("*  loose  -current [{}, {}]  -default [{}, {}]".format(mg.gt_const[1][0], mg.gt_const[1][1],
#                                                                       mg.gt_const[1][2], mg.gt_const[1][3]))
#         print("*Rotation Gain:")
#         print("*  tight  -current [{}, {}]  -default [{}, {}]".format(mg.gr_const[0][0], mg.gr_const[0][1],
#                                                                       mg.gr_const[0][2], mg.gr_const[0][3]))
#         print("*  loose  -current [{}, {}]  -default [{}, {}]".format(mg.gr_const[1][0], mg.gr_const[1][1],
#                                                                       mg.gr_const[1][2], mg.gr_const[1][3]))
#         print("*Curvature Gain:")
#         print("*  tight  -current [{}, {}]  -default [{}, {}]".format(mg.gc_const[0][0], mg.gc_const[0][1],
#                                                                       mg.gc_const[0][2], mg.gc_const[0][3]))
#         print("*  loose  -current [{}, {}]  -default [{}, {}]".format(mg.gc_const[1][0], mg.gc_const[1][1],
#                                                                       mg.gc_const[1][2], mg.gc_const[1][3]))
#         print("*Bend Gain:")
#         print(
#             "*  tight  -current [{}, {}]  -default [{}, {}]".format(mg.gb_const[0][0], mg.gb_const[0][1],
#                                                                     mg.gb_const[0][2], mg.gb_const[0][3]))
#         print(
#             "*  loose  -current [{}, {}]  -default [{}, {}]".format(mg.gb_const[1][0], mg.gb_const[1][1],
#                                                                     mg.gb_const[1][2], mg.gb_const[1][3]))
#         print("*Primary Rdw Gain Rates")
#         print("*  gt changing rate - {}".format(mg.g_rate_const[0][0]))
#         print("*  gr changing rate - {}".format(mg.g_rate_const[0][1]))
#         print("*  gc changing rate - {}".format(mg.g_rate_const[0][2]))
#         print("*  gb changing rate - {}".format(mg.g_rate_const[0][3]))
#         print("*Primary Rdw Gain Rates")
#         print("*  gt primary value - {}, v_vir / v_phy = {}".format(mg.g_pri[0][0], mg.g_pri[0][0]))
#         print("*  gr primary value - {}, r_rot / r_phy = {}".format(mg.g_pri[0][1], mg.g_pri[0][1]))
#         radius = 'inf' if mg.g_pri[0][2] == 0 else str(1 / mg.g_pri[0][2])
#         print("*  gc primary value - {}, radius_phy = {}".format(mg.g_pri[0][2], radius))
#         print("*  gb primary value - {}, radius_vir / radius_phy = {}".format(mg.g_pri[0][3],
#                                                                               mg.g_pri[0][3]))
#         print("*----------------------------------------------------------------")
#
#     def log_steer_paras(self):
#         print("*-------------------------Steer info-----------------------------")
#         print("*  rdw controller type - " + str(type(self.rdw_env.rdwer).__name__))
#         print("*  steer apf repulsion factor - {}".format(self.rdw_env.rdwer.apf_force_c[0]))
#         print("*  steer apf gravitation factor - {}".format(self.rdw_env.rdwer.apf_force_c[1]))
#         print("*  steer dampening distance - {} m".format(self.rdw_env.rdwer.steer_dampen[0]))
#         print("*  steer dampening bearing - {} deg".format(self.rdw_env.rdwer.steer_dampen[1]))
#         print("*  moving down threshold - {} m/s".format(self.rdw_env.rdwer.steer_vel_c[1]))
#         print("*  max moving rotation - {} deg/s".format(self.rdw_env.rdwer.steer_vel_c[2]))
#         print("*  max head rotation - {} deg/s".format(self.rdw_env.rdwer.steer_vel_c[3]))
#         print("*-----------------------------------------------------------------")
#
#     def log_reset_paras(self):
#         print("*-------------------------Reset info-----------------------------")
#         print("*  reset controller type - " + str(type(self.rdw_env.resetter).__name__))
#         print("*  reset trigger threshold - {} m".format(self.rdw_env.resetter.reset_trigger_t))
#         print("*  reset cancel angle threshold - {} deg".format(self.rdw_env.resetter.reset_terminal_t))
#         print("*-----------------------------------------------------------------")
#
#     def log_simuwalker_paras(self):
#         print("*-------------------------Simulation info-----------------------------")
#         print("*  walk type - " + str(type(self.rdw_env.inputer).__name__))
#         print("*  min walking speed - {} m/s".format(self.rdw_env.inputer.mv_vel_range[0]))
#         print("*  normal walking speed - {} m/s".format(self.rdw_env.inputer.norm_phy_vels[0]))
#         print("*  max walking speed - {} m/s".format(self.rdw_env.inputer.mv_vel_range[1]))
#         print("*  normal rotate speed - {} deg/s".format(self.rdw_env.inputer.norm_phy_vels[1]))
#         print("*  max rotate speed - {} deg/s".format(self.rdw_env.inputer.rot_vel_range[1]))
#         print("*  target end rotation threshold - {} deg".format(self.rdw_env.inputer.end_motion_thr[1]))
#         print("*  target end moving threshold - {} m".format(self.rdw_env.inputer.end_motion_thr[0]))
#         print("*-----------------------------------------------------------------")
#
#     def log_vir_patches_matrix(self):
#         print("patches connection matrix")
#         head = "  "
#         for i in range(len(self.rdw_env.inputer.patches_mat)):
#             head += str(i) + "  "
#         print(head)
#         j = 0
#         for a in self.rdw_env.inputer.patches_mat:
#             body = str(j) + " "
#             if j < 10:
#                 body += " "
#             for data in a:
#                 body += str(data) + "  "
#             print(body)
#             j += 1
#         for i in range(len(self.rdw_env.inputer.patches_mat)):
#             l = self.rdw_env.inputer.find_nei_patches(i)
#             print("patch id: ", i, " connected patch", l)
#
#     def log_recorded_patches_lists(self):
#         if len(self.rdw_env.inputer.patches_lists) == 0:
#             print("no existed data")
#         for patch_list in self.rdw_env.inputer.patches_lists:
#             print("recorded patch:", patch_list)
#
#     def log_recorded_targets_lists(self):
#         if len(self.rdw_env.inputer.recorded_walking_targets_x_lists) == 0:
#             print("no existed data")
#         for i in range(len(self.rdw_env.inputer.recorded_walking_targets_x_lists)):
#             print("recorded patch:", self.rdw_env.inputer.patches_lists[i])
#             str_info = "recorded targets:\n"
#             target_x_list = self.rdw_env.inputer.recorded_walking_targets_x_lists[i]
#             target_y_list = self.rdw_env.inputer.recorded_walking_targets_y_lists[i]
#             for j in range(len(target_x_list)):
#                 target_x = target_x_list[j]
#                 target_y = target_y_list[j]
#                 str_info += str(target_x) + " " + str(target_y) + "\n"
#             print(str_info)
