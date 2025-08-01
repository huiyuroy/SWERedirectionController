import numpy as np
import torch

DEFAULT_AGENT_CONFIG = {
    'base': {
        'gamma': 0.99,  # Discounted Factor
        'tau': 0.005  # polyak update source network to target network, =1 means totally eval net -> target net
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

DEFAULT_MEMORY_CONFIG = {
    's': [(-1,), np.float32],
    'a': [(-1,), np.float32],
    'r': [(1,), np.float32],
    's_': [(-1,), np.float32],
    'dw': [(1,), np.bool_]
}

DEFAULT_TRAIN_CONFIG = {
    'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),  # cuda or cpu
    'max_train_step': int(5e6),  # Max steps of entire training
    'max_epi_step': 2000,  # Max steps of one single episode
    'update_step': 50,  # training frequency, in steps
    'save_step': int(1e5),  # Model saving interval, in steps.
    'eval_step': int(5e3),  # Model evaluating interval, in steps.
    'enable_random_seed': True,
    'random_seed': 0
}



