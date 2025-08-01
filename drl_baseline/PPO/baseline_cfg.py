from ..common_cfg import *

PPO_BASELINE_CONFIG = {
    'base': {
        'gamma': 0.99,  # Discounted Factor
        'tau': 0.005,  # polyak update source network to target network, =1 means totally eval net -> target net
        'k_epochs': 10,  # PPO update times
        'lambd': 0.95,  # GAE Factor
        'clip_rate': 0.2,  # PPO Clip rate
        'l2_reg': 1e-3,  # L2 regulization coefficient for Critic
        'actor_lr': 1e-4,  # Learning rate of actor
        'critic_lr': 1e-4,  # Learning rate of critic
        'actor_mini_batch': 64,  # lenth of sliced trajectory of actor
        'critic_mini_batch': 64,  # lenth of sliced trajectory of actor
        'entropy_coef': 0.01,  # Entropy coefficient of Actor
        'entropy_coef_decay': 0.99  # Decay rate of entropy_coef
    },
    'memory': {
        'max_size': int(1e6),  # as batch size for ppo
        'batch_size': 256,  # as mini batch for ppo
        "alpha": 0.6,  # used for prior memory
        "epsilon": 0.0001,  # used for prior memory
        "use_latest": False,
        'prior_enable': False
    },
    'env': {
        "action_dim": None,
        "state_dim": None,
        "frame_dim": [
            None,
            None,
            None
        ],
        "action_shape": [None, None]  # max, min for continuous
    }
}

PPO_BASE_MEMORY_CONFIG = {
    's': [(-1,), np.float32],
    'a': [(-1,), np.float32],
    'r': [(1,), np.float32],
    'a_logprob': [(-1), np.float32],
    's_': [(-1,), np.float32],
    'done': [(1,), np.bool_],
    'dw': [(1,), np.bool_]
}

PPO_BASE_TRAIN_CONFIG = {
    'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),  # cuda or cpu
    'max_train_step': int(5e6),  # Max steps of entire training
    'max_epi_step': 2000,  # Max steps of one single episode
    'update_step': 50,  # training frequency, in steps
    'save_step': int(1e5),  # Model saving interval, in steps.
    'eval_step': int(5e3),  # Model evaluating interval, in steps.
    'enable_random_seed': True,
    'random_seed': 0
}
