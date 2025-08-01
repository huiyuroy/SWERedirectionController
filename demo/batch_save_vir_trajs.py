import gc
import os

import pyrdw.generator as generator
from common import data_path
from pyrdw.util.simu_traj_gen.base_traj import gen_simu_tiling_rand_trajs, gen_simu_abs_rand_trajs, gen_simu_road_trajs

if __name__ == '__main__':
    v_path = data_path + '\\vir'
    v_files = [os.path.join(v_path, file) for file in os.listdir(v_path) if 'json' in file]

    for v_scene_path in v_files:
        # 4.加载虚拟空间,这里批量化加载，也可加载一个
        _, v_scene_name = os.path.split(v_scene_path)
        v_name = v_scene_name.split(".")[0]
        vscene = generator.load_scene(v_path + '\\' + v_name + '.json', load_road=True)
        vscene.update_grids_runtime_attr()

        abs_road_trajs = gen_simu_road_trajs(vscene, 10)
        for i, traj in enumerate(abs_road_trajs):
            generator.save_trajectory(traj, v_path + '\\simu_trajs\\{}\\{}.json'.format(v_name, 'abs_road_' + str(i)))
        print('abs_road_done!')

        proxy_road_trajs = gen_simu_road_trajs(vscene, 10, True)
        for i, traj in enumerate(proxy_road_trajs):
            generator.save_trajectory(traj, v_path + '\\simu_trajs\\{}\\{}.json'.format(v_name, 'prox_road_' + str(i)))
        print('prox_road_done!')

        abs_rand_trajs = gen_simu_abs_rand_trajs(vscene, 10)
        for i, traj in enumerate(abs_rand_trajs):
            generator.save_trajectory(traj, v_path + '\\simu_trajs\\{}\\{}.json'.format(v_name, 'abs_rand_' + str(i)))
        print('abs_rand_done!')

        tiling_rand_trajs = gen_simu_tiling_rand_trajs(vscene, 10)
        for i, traj in enumerate(tiling_rand_trajs):
            generator.save_trajectory(traj,
                                      v_path + '\\simu_trajs\\{}\\{}.json'.format(v_name, 'tiling_rand_' + str(i)))
        print('tiling_rand_done!')
