import os
import numpy as np
import pyrdw.lib.math.geometry as geo
import pyrdw.generator as generator
import pyrdw.default as default
from common import data_path
from pyrdw.core.env.base import RdwEnv


"""
To enable mouse-keyboard simulation, use mouse drag to rotate user's direction, and use w-a-s-d control approach to 
manipulate user position
"""

if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find('pyrdw') + len('pyrdw')]
    # 1.创建环境
    env = RdwEnv()

    # 2.添加agents，必须在env创建后调用，例子中加载了no+r21和no+r2mpe算法
    env.add_agent(default.obtain_agent(gainer='arc', rdwer='arc', resetter='rarc', inputer='live'),
                  name='arc')
    # env.add_agent(generator.obtain_agent(gainer='simple', rdwer='apfs2t', resetter='apfs2t', inputer='live'),
    #               name='apfs2t')
    env.load_logger()  # 加载日志记录器，可以自己设计
    env.load_ui()  # 加载ui界面，可以自己设计

    # 3.加载物理空间
    pname = 'test2'
    pscene = generator.load_scene(data_path + '\\phy\\' + pname + '.json')
    pscene.update_grids_runtime_attr()

    # pscene.calc_r2mpe_vis_range_simplify()
    # pscene.calc_apfs2t_precomputation()  # 如任何组件使用apfs2t，则必须调用此方法

    # 4.加载虚拟空间
    vname = 'abnormal_s5'
    vscene = generator.load_scene(data_path + '\\vir\\' + vname + '.json')
    vscene.update_grids_runtime_attr()
    # 5.设置环境虚拟和真实空间
    env.set_scenes(vscene, pscene)  # 设置虚拟和物理空间，必须在prepare前调用

    # 6.为输入管理器提供输入回调，demo中采用ui的键鼠监听作为

    # 7.env组件预处理，如果更换场景或者更换模拟路径后必须调用
    env.prepare()  # 加载预处理内容，例如场景可见性划分、离散占位网格处理等
    env.env_ui.render_mode(True)

    max_area_conv = None
    max_area = 0
    for conv in env.p_scene.conv_polys:
        conv_area = geo.calc_poly_area(np.array(conv.vertices))
        if conv_area > max_area:
            max_area = conv_area
            max_area_conv = conv
    init_p_loc = max_area_conv.center

    max_area_conv = None
    max_area = 0
    for conv in env.v_scene.conv_polys:
        conv_area = geo.calc_poly_area(np.array(conv.vertices))
        if conv_area > max_area:
            max_area = conv_area
            max_area_conv = conv
    init_v_loc = max_area_conv.center

    env.init_agents_state(p_loc=init_p_loc, p_fwd=[0, 1], v_loc=init_v_loc.tolist(), v_fwd=[0, 1])

    env.reset()
    while True:
        env.receive()
        d = env.step()
        env.render()

        if d:
            break
