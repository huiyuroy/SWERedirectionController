const_env = {
    "tiling_width": 10,
    "time_step": 0.02,
    "human_radius": 20,  # 默认人体平均宽度
    "scene_board": 40,  # 绘制场景时在长宽添加的少量外边缘宽度，防止场景空白区域与背景混合，默认0.4m
    "file_scene_scale": 1  # 场景数值与实际单位的换算关系，1
}

const_steer = {
    "static_rot": 4.0,
    "move_dt": 20,
    "rot_dt": 1.5,
    "max_gc_rot": 15.0,
    "max_gr_rot": 30.0,
    "dampen_dis": 125,
    "dampen_bear": 45.0,
    "apf_rep_ft": 1.0,
    "apf_gra_ft": 1.0
}

const_gain = {
    "trans_gain": [
        [
            0.86,
            1.26,
            0.86,
            1.26
        ],  # low, high, default_low, default_high
        [
            0.6,
            1.8,
            0.6,
            1.8
        ]
    ],
    "rot_gain": [
        [
            0.8,
            1.49,
            0.8,
            1.49
        ],  # low, high, default_low, default_high
        [
            0.45,
            3.25,
            0.45,
            3.25
        ]
    ],
    "cur_gain": [
        [
            -0.045,
            0.045,
            -0.045,
            0.045
        ],  # low, high, default_low, default_high
        [
            -0.2,
            0.2,
            -0.2,
            0.2
        ]
    ],
    "bend_gain": [
        [
            1.0,
            1.0,
            1.0,
            1.0
        ],
        [
            1.0,
            1.0,
            1.0,
            1.0
        ]
    ],
    "gains_rate": [
        [
            0.0,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0
        ]
    ],
    "pri_gains": [
        [
            1.0,  # translation gain
            1.0,  # rotation gain
            0.133333333,  # curvature gain 1/7.5 m-1
            4.0
        ],
        [
            1.0,
            1.0,
            0.0,
            4.0
        ]
    ]
}

const_reset = {
    "reset_trigger_dis": 20,
    "reset_finish_ang": 1,  # in degree
    "reset_pred_dis": 20
}

const_simu = {
    "human_step": 120,
    "walk_vel_range": [
        20,
        200,
        20,
        200
    ],
    "norm_walk_vel": [
        125,
        125
    ],
    "rotate_vel_range": [
        0,
        180,
        0,
        180
    ],
    "norm_rot_vel": [
        45,
        45
    ],
    "end_motion_thresholds": [
        10,
        1.0,
        10,
        1.0
    ]
}
