import copy
import datetime
import os

import numpy as np

import pyrdw.lib.math.geometry as geo
import pyrdw.generator as generator
import rlrdw.rlrdw_generator as r_gen
from rlrdw.swerc.swerc_benchmark import SWERCDiscreteScene
from rlrdw.swerc.swerc_trainer import SWERCRdwEnv
from rlrdw.training_scenes import abbr_name
from common import data_path, project_path


if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find('pyrdw') + len('pyrdw')]
    # 1. create env
    env = SWERCRdwEnv()
    # 2.add rdw agents, must be called after env
    pname = 'mess'
    vname = 'complex2000x2000'
    save_name = '{}_{}'.format(abbr_name[pname], abbr_name[vname])

    env.add_agent(r_gen.obtain_agent(inputer='traj', agent_manager='swerc', swerc_reset_type='normal'), name='swerc')
    env.agents['swerc'].load('{}_{}_'.format(abbr_name[pname], abbr_name[vname]))

    env.load_logger()  # load log
    env.load_ui('swerc test')  # load ui

    # 3.load phy spaces
    p_path = data_path + '\\rl\\phy'
    pscene: SWERCDiscreteScene = r_gen.load_scene(p_path + '\\' + pname + '.json', scene_class='swerc')
    pscene.update_grids_runtime_attr()
    pscene.calc_swe_energy()

    # v_path = data_path + '\\vir'
    v_path = data_path + '\\rl\\vir'
    # v_files = [os.path.join(v_path, file) for file in os.listdir(v_path) if 'json' in file]
    v_files = [vname]
    for v_scene_path in v_files:
        # 4. load vir spaces
        _, v_scene_name = os.path.split(v_scene_path)
        v_name = v_scene_name.split(".")[0]
        vscene = r_gen.load_scene(v_path + '\\' + v_name + '.json', load_extend=True)
        vscene.update_grids_runtime_attr()

        # 5.setup virtual and physical spaces
        env.set_scenes(vscene, pscene)  # 设置虚拟和物理空间，必须在prepare前调用

        # 6. load virtual trajectories
        vtrajs = generator.load_trajectories(v_path, v_name)
        vtrajs = vtrajs[::10]

        for vtraj in vtrajs:

            vtraj.range_distance(20000)  # set walking distance to 200m
        env.set_trajectories(vtrajs)

        # 7.env prepare
        env.prepare()  # load pre-computation components
        env.env_ui.render_mode(True)

        # 8. main loop
        for traj_idx, traj in enumerate(vtrajs):
            env.set_current_trajectory(traj)
            max_area_conv = None
            max_area = 0
            for conv in env.p_scene.conv_polys:
                conv_area = geo.calc_poly_area(np.array(conv.vertices))
                if conv_area > max_area:
                    max_area = conv_area
                    max_area_conv = conv
            init_p_loc = max_area_conv.center
            env.init_agents_state(p_loc=init_p_loc, p_fwd=[0, 1], v_loc=[0, 1], v_fwd=[0, 1])
            env.reset()
            while True:
                done = env.step()
                env.render()
                env.record()
                if done:
                    all_data = env.output_epi_info()
                    d = datetime.datetime.now()
                    print('tid:{}'.format(traj_idx), end=', ')
                    for data in all_data:
                        print('{alg_name}: [resets {reset_num}/{total_reset_num}'
                              ' rdw rate {mean_rdw_rate:.2f} '
                              'avg dis btw resets {avg_dis_btw_resets:.2f}]'.format(**data), end=', ')
                    print()
                    break
