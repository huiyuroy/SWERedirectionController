from typing import Tuple, Sequence

import numpy as np
import torch


def compute_output_spatial_shape(
        input_shape: Tuple[int, int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int] | None = None,
        dilations: Sequence[int] | None = None,
) -> Tuple[int, int]:
    """
    Calculates the output height and width based on the input
    height and width of the convolution layer.

    ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d

    Args:
        input_shape: [h,w]
        kernel_sizes: ordered by following cnn layers
        strides: ordered by following cnn layers
        paddings: ordered by following cnn layers
        dilations: ordered by following cnn layers

    Returns:

    """
    output_shape = list(input_shape)
    cnn_stack_len = len(kernel_sizes)
    if paddings is None:
        paddings = [0] * cnn_stack_len
    if dilations is None:
        dilations = [1] * cnn_stack_len

    for i in range(len(kernel_sizes)):
        for j in range(len(input_shape)):
            output_shape[j] = int(
                np.floor(
                    ((output_shape[j] + 2 * paddings[i] - dilations[i] * (kernel_sizes[i] - 1) - 1) / strides[i]) + 1)
            )

    return tuple(output_shape)  # type: ignore


def epsilon_greedy_value(eps_init, eps_end, total_decay_steps, cur_step):
    fraction = min(float(cur_step) / total_decay_steps, 1.0)
    return eps_init + fraction * (eps_end - eps_init)


def action_mapping(a, act_min, act_max):
    return (a + 1) * 0.5 * (act_max - act_min) + act_min


# def continuous_action_mapping(agent_action, env_action_shape):
def gym_reward_adapter(r, env_name, **kwargs):
    # For Pendulum-v0
    if 'Pendulum' in env_name:
        r = (r + 8) / 8
    # For LunarLander
    elif 'LunarLander' in env_name:
        if r <= -100:
            r = -10
    # For BipedalWalker
    elif 'BipedalWalker' in env_name:
        if r <= -100:
            r = -1
    elif 'MountainCarContinuous' in env_name:
        s = kwargs['s']
        s_ = kwargs['s_']
        done = kwargs['done']
        # if abs(s_[1] == 0):
        #     r = 0
        # else:
        #     if s_[1] < 0:
        #         r = -1
        #     elif s_[0] > -0.5 and s_[1] > 0:
        #         r = 1
        r = s_[0] - 0.4

        if done:
            r = 100

    return r


def understand_gather():
    """
    torch.gather function, parameter list: input, dim, index, *, sparse_grad=False, out=None
    - The output tensor has the same shape as the index tensor
    - dim represents that the data in index directly replaces the index of a certain dim in input, for example dim=0, index=[[2,1,0]]
    - The output is output[[input[2,j1], input[1,j2], input[0,j3]]]
    - j1,j2,j3 are determined by the indices of 2,1,0 in index. The index of 2 in index is (0,1). Since 2 is already used to represent the 0th dim of output, we select the 1st dim of index' to denote j1
    - Therefore j1,j2,j3 are 0,1,2 respectively, resulting in output[[input[2,0], input[1,1], input[0,2]]]


    Returns:

    """
    t = torch.arange(1, 7).view(2, 3)
    print(t)
    index = t.argmax(0, keepdim=True)

    print(index)
    t1 = torch.arange(1, 7).view(2, 3)
    print(t1.gather(0, index))
