import copy

import numpy as np

from drl_baseline.PPO.baseline_cfg import PPO_BASE_TRAIN_CONFIG, PPO_BASELINE_CONFIG, PPO_BASE_MEMORY_CONFIG


lidar_num = 18
# ploc, pfwd, pvel, prot, vloc, vfwd, vvel, vrot, reset_state, pdis2every_obs, vdis2every_obs
swerc_state_dim = 2 + 2 + 2 + 2 + lidar_num + lidar_num
swerc_img_state_dim = (1, 67, 67)  # 4 channel, phy features, 67 is to make sure the cnn output fits powers of 2.
swerc_state_seq = 10
swerc_action_dim = 3  # gt, gr, gc, steer rot
swerc_action_repeat = 5
swerc_max_v_dis = 20000  # 200m maximum virtual walking distance

# ---------------------------------------------------------------------------------------------------------------------
SWERCPPO_CONFIG = copy.deepcopy(PPO_BASELINE_CONFIG)
SWERCPPO_CONFIG['base']['entropy_coef'] = 0.02
SWERCPPO_CONFIG['base']['actor_lr'] = 1e-4
SWERCPPO_CONFIG['base']['critic_lr'] = 1e-4
SWERCPPO_CONFIG['memory']['max_size'] = 2048
SWERCPPO_CONFIG['memory']['batch_size'] = 128
SWERCPPO_CONFIG['env']['action_dim'] = swerc_action_dim
SWERCPPO_CONFIG['env']['state_dim'] = swerc_state_dim
SWERCPPO_CONFIG['env']['img_state_dim'] = swerc_img_state_dim

SWERCPPO_MEMORY_CONFIG = copy.deepcopy(PPO_BASE_MEMORY_CONFIG)
SWERCPPO_MEMORY_CONFIG['s'][0] = (swerc_state_dim,) if swerc_state_seq == 1 else (swerc_state_seq, swerc_state_dim)
SWERCPPO_MEMORY_CONFIG['a'][0] = (swerc_action_dim,)
SWERCPPO_MEMORY_CONFIG['a_logprob'] = [(swerc_action_dim,), np.float32]
SWERCPPO_MEMORY_CONFIG['r'][0] = (1,)
SWERCPPO_MEMORY_CONFIG['s_'][0] = (swerc_state_dim,) if swerc_state_seq == 1 else (swerc_state_seq, swerc_state_dim)
SWERCPPO_MEMORY_CONFIG['dw'] = [(1,), np.bool_]
SWERCPPO_MEMORY_CONFIG['done'] = [(1,), np.bool_]

SWERCPPO_TRAIN_CONFIG = copy.deepcopy(PPO_BASE_TRAIN_CONFIG)
SWERCPPO_TRAIN_CONFIG['max_epi_step'] = 9000  # follow the original paper
SWERCPPO_TRAIN_CONFIG['max_train_step'] = int(1.8e6)  # at least 100,000 epoches
SWERCPPO_TRAIN_CONFIG['learn_step'] = 50
SWERCPPO_TRAIN_CONFIG['eval_step'] = int(50e3)  # turn off this para in drl_rdw benchmark
SWERCPPO_TRAIN_CONFIG['save_step'] = int(9e4)
SWERCPPO_TRAIN_CONFIG['eval_epi'] = 100  # evaluate every 100 epoches

# ---------------------------------------------------------------------------------------------------------------------