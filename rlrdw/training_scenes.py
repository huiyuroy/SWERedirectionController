import gc
import math
import random
import numpy as np
from shapely import Point
from shapely.geometry.linestring import LineString

import pyrdw.generator as generator
import pyrdw.lib.math.geometry as geo
import pyrdw.lib.math.algebra as alg

from pyrdw.core import DEG2RAD
from pyrdw.core.space.scene import DiscreteScene
from pyrdw.core.space.trajectory import Trajectory

from common import data_path


def generate_srl_training_scene():
    phy_path = data_path + '\\rl\\srl\\phy'
    name = 'empty579x579'
    contour_list = [[[0.0, 579.0], [579.0, 579.0], [579.0, 0.0], [0.0, 0.0]]]
    scene = DiscreteScene()
    scene.update_contours(name, contour_list)
    scene.update_segmentation()
    scene.update_grids_precompute_attr(True, True, True)
    generator.save_scene(scene, phy_path)

    vir_path = data_path + '\\rl\\srl\\vir'
    name = 'empty20000x20000'
    contour_list = [[[0.0, 20000.0], [20000.0, 20000.0], [20000.0, 0.0], [0.0, 0.0]]]
    scene = DiscreteScene()
    scene.update_contours(name, contour_list)
    scene.update_segmentation()
    scene.update_grids_precompute_attr(False, False, False)
    generator.save_scene(scene, vir_path)


def generate_s2ot_training_scene():
    phy_path = data_path + '\\rl\\s2ot\\phy'
    name = 'empty400x400'
    contour_list = [[[0.0, 400.0], [400.0, 400.0], [400.0, 0.0], [0.0, 0.0]]]
    scene = DiscreteScene()
    scene.update_contours(name, contour_list)
    scene.update_segmentation()
    scene.update_grids_precompute_attr(True, True, True)  # we don't need to compute visible grid occupancy
    generator.save_scene(scene, phy_path)

    vir_path = data_path + '\\rl\\s2ot\\vir'
    name = 'empty2000x2000'
    contour_list = [[[0.0, 2000.0], [2000.0, 2000.0], [2000.0, 0.0], [0.0, 0.0]]]
    scene = DiscreteScene()
    scene.update_contours(name, contour_list)
    scene.update_segmentation()
    scene.update_grids_precompute_attr(True, False, True)  # we don't need to compute visible grid occupancy
    generator.save_scene(scene, vir_path)


def generate_src_training_pscenes():
    w = 1000  # original work use 10 x 10 phy spaces
    names = ('complex', 'simple', 'empty',)
    contours = (
        [
            [

                [w, 0],
                [w, w],
                [3 * w / 4, w],
                [3 * w / 4, 2 * w / 3],
                [w / 4, 2 * w / 3],
                [w / 4, w],
                [0, w], [0, 3 * w / 8],
                [w / 8, 3 * w / 8],
                [w / 8, 0],
            ],
            [
                [5 * w / 6, w / 6],
                [5 * w / 6, w / 3],
                [2 * w / 3, w / 3],
                [2 * w / 3, w / 6]
            ]

        ],
        [[[0, 0],
          [w, 0],
          [w, w],
          [3 * w / 4, w],
          [3 * w / 4, 2 * w / 3],
          [w / 4, 2 * w / 3],
          [w / 4, w],
          [0, w]]],  # contour for simple
        [[[0.0, w], [w, w], [w, 0.0], [0.0, 0.0]]],  # contour for empty
    )
    phy_path = data_path + '\\rl\\src\\phy'

    for name, contour in zip(names, contours):
        scene = DiscreteScene()
        scene.update_contours(name, contour)
        scene.update_segmentation()
        scene.update_grids_precompute_attr(True, True, True)
        generator.save_scene(scene, phy_path)


def generate_src_training_vscenes():
    def __generate_src_vir_obs_contours(obs_grid_w=960.0, vir_w=5000.0):
        """

        Args:
            obs_grid_w: 每个包含障碍物网格的宽度
            vir_w: 虚拟空间总宽度

        Returns:

        """
        vir_free_w = vir_w - 2 * 100  # safe bound width
        grid_num = int(vir_free_w / obs_grid_w)
        vir_free_origin = np.array([100, 100])
        obs = []

        for i in range(grid_num):
            for j in range(grid_num):
                if np.random.randint(0, 3) != 1:  # 50% chance to place an obstacle in the grid
                    grid_rect_lb = vir_free_origin + np.array([i * obs_grid_w, j * obs_grid_w])
                    grid_rect_rt = vir_free_origin + np.array([(i + 1) * obs_grid_w, (j + 1) * obs_grid_w])
                    grid_rect = np.array([
                        grid_rect_lb,
                        np.array([grid_rect_rt[0], grid_rect_lb[1]]),
                        grid_rect_rt,
                        np.array([grid_rect_lb[0], grid_rect_rt[1]])
                    ])
                    grid_xmin, grid_xmax = grid_rect_lb[0], grid_rect_rt[0]
                    grid_ymin, grid_ymax = grid_rect_lb[1], grid_rect_rt[1]

                    find = False
                    obs_contour = []
                    while not find:
                        obs_radius = np.random.randint(50, 200)
                        obs_center = grid_rect_lb + np.random.rand(2) * obs_grid_w * 0.5
                        obs_xmin, obs_xmax = obs_center[0] - obs_radius, obs_center[0] + obs_radius
                        obs_ymin, obs_ymax = obs_center[1] - obs_radius, obs_center[1] + obs_radius

                        if (grid_xmin <= obs_xmin <= grid_xmax and grid_xmin <= obs_xmax <= grid_xmax
                                and grid_ymin <= obs_ymin <= grid_ymax and grid_ymin <= obs_ymax <= grid_ymax):
                            find = True
                            n = np.random.randint(3, 5)
                            thetas = list(np.random.rand(n) * (2 * np.pi))
                            thetas.sort()
                            for t in thetas:
                                obs_contour.append((obs_center + np.array([obs_radius * math.cos(t),
                                                                           obs_radius * math.sin(t)])).tolist())

                    obs.append(obs_contour)
                print('\rcontour generating: {:.2f}%'.format((i * grid_num + j + 1) / (grid_num ** 2) * 100), end='')
        print()
        return obs

    names = ('complex2000x2000',)
    complex_w = 2000.0
    complex_contour = [[[0.0, complex_w], [complex_w, complex_w], [complex_w, 0.0], [0.0, 0.0]]]
    obs = __generate_src_vir_obs_contours(obs_grid_w=400, vir_w=complex_w)
    complex_contour.extend(obs)
    contours = (
        # [[[0.0, 1000.0], [1000.0, 1000.0], [1000.0, 0.0], [0.0, 0.0]]],
        # [[[0.0, 5000.0], [5000.0, 5000.0], [5000.0, 0.0], [0.0, 0.0]]],  # contour for 5000x5000
        complex_contour,
    )
    vir_path = data_path + '\\rl\\vir'
    for name, contour in zip(names, contours):
        scene = DiscreteScene()
        scene.update_contours(name, contour)
        scene.update_segmentation()
        scene.update_grids_precompute_attr(True, False, True)
        generator.save_scene(scene, vir_path)


def generate_src_vir_trajs():
    def __obtain_src_random_traj(time_step, vir_scene, simu_num=int(1e4)):
        """

        Args:

            time_step: 0.02s default
            vir_scene:
            tar_distribute: 行列目标细分数量，
            simu_num: number of simulate trajectories.

        Returns:

        """
        max_travel_dis = time_step * 18000 * 200
        dis_split = 25
        trajs = []

        w, h = vir_scene.tilings_shape

        for i in range(simu_num):
            cur_tiling = None
            init_done = False
            while not init_done:
                pot_ = vir_scene.tilings_walkable[np.random.randint(0, len(vir_scene.tilings_walkable))]
                if pot_.type:
                    cur_tiling = pot_
                    init_done = True
            cur_pos = cur_tiling.center
            done = False
            traveled_dis = 0
            traj_tars = []
            while not done:
                r, c = cur_tiling.mat_loc
                find = False
                nxt_id = (c, r)
                while not find:
                    search_range = np.random.randint(200, 601)
                    cir_bound = np.array(Point(c, r).buffer(int(search_range / vir_scene.tiling_w)).exterior.coords,
                                         dtype=np.int32)
                    pot_tid = []
                    for c_id in cir_bound:
                        if 0 <= c_id[0] < w and 0 <= c_id[1] < h:
                            pot_tiling = vir_scene.tilings[c_id[0] + c_id[1] * w]
                            pot_pos = pot_tiling.center
                            if pot_tiling.type and vir_scene.poly_contour.covers(LineString((cur_pos, pot_pos))):
                                pot_tid.append(c_id.tolist())
                    if len(pot_tid) > 0:
                        nxt_id = random.choice(pot_tid)
                        find = True
                next_tiling = vir_scene.tilings[nxt_id[0] + nxt_id[1] * w]
                next_pos = next_tiling.center
                fwd = next_pos - cur_pos
                n_fwd = geo.norm_vec(fwd)
                dis_bias = alg.l2_norm(fwd)
                split_time = int(dis_bias / dis_split)
                if split_time > 0:
                    delta_dis = dis_bias / split_time
                    for k in range(split_time):
                        traj_tars.append(n_fwd * k * delta_dis + cur_pos)
                else:
                    traj_tars.append(cur_pos)
                cur_pos = next_pos
                cur_tiling = next_tiling
                traveled_dis += dis_bias
                if traveled_dis >= max_travel_dis:
                    done = True

            t = Trajectory()
            t.type = 'grid_rand'
            t.tar_data = traj_tars
            t.tar_num = len(t.tar_data)
            t.range_targets()
            trajs.append(t)

            print('\rtraj generate in {:.2f}%'.format((i + 1) / simu_num * 100), end='')
        print()
        return trajs

    for v_name in ('complex2000x2000','complex5000x5000'):
        v_path = data_path + '\\rl\\vir'
        vscene = generator.load_scene(v_path + '\\' + v_name + '.json')
        vscene.update_grids_runtime_attr()
        trajs = __obtain_src_random_traj(0.02, vscene, 5000)
        for i, traj in enumerate(trajs):
            for t in traj.tar_data:
                t_tiling, _ = vscene.calc_located_tiling_conv(np.array(t))
                if not t_tiling.type and len(t_tiling.cross_bound) == 0:
                    raise Exception('traj tar wrong, not within the scene')
            generator.save_trajectory(traj, v_path + '\\simu_trajs\\{}\\{}.json'.format(v_name, 'traj' + str(i)))
            print('\rchecking and saving traj rationality: {:.2f}%'.format((i + 1) / len(trajs) * 100), end='')
        print()


def generate_srl_vir_trajs():
    def __obtain_srl_random_traj(time_step, vir_shape, simu_num=int(1e4)):
        """

        Args:
            time_step: 0.02s default
            vir_shape: [x_max, y_max]
            simu_num: number of simulate trajectories.

        Returns:

        """
        x_min, y_min = 0, 0
        x_max, y_max = vir_shape

        # just much longer than max simulation walking distance (1.4m/s x 60s (3600 fps))
        max_travel_dis = 140 * time_step * 5000
        split_dis = 25.0  # cm
        curve_split_dis = 5
        trajs = []

        for i in range(simu_num):
            cur_pos = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
            cur_fwd = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            if alg.l2_norm(cur_fwd) == 0:
                cur_fwd = np.array([1, 0])
            cur_fwd = geo.norm_vec(cur_fwd)

            done = False
            turning = False
            traveled_dis = 0
            traj_tars = []
            while not done:
                if not turning:  # walk straight
                    split_time = np.random.randint(2, 15)
                    dis = split_time * split_dis  # distance between 50-350 cm
                    offsets = [cur_fwd.copy() * t for t in range(split_time)]
                    traj_tars.extend(list(map(lambda x: x * split_dis + cur_pos, offsets)))
                    cur_pos = dis * cur_fwd + cur_pos
                    turning = True
                else:
                    radius = np.random.uniform(400, 800)
                    theta = np.random.normal(0, 22.5 * DEG2RAD)
                    dis = abs(radius * theta)
                    split_time = math.floor(dis / curve_split_dis)
                    if split_time > 0:
                        delta_dis = dis / split_time
                        delta_theta = theta / split_time
                        for _ in range(split_time):
                            cur_fwd = geo.rot_vecs(cur_fwd, delta_theta)
                            cur_pos = cur_fwd * delta_dis + cur_pos
                            traj_tars.append(cur_pos.copy())
                    else:
                        cur_fwd = geo.rot_vecs(cur_fwd, theta)
                        cur_pos = cur_fwd * dis + cur_pos
                        traj_tars.append(cur_pos.copy())
                    turning = False

                traveled_dis += dis
                if traveled_dis >= max_travel_dis:
                    done = True

            t = Trajectory()
            t.tar_data = traj_tars
            t.tar_num = len(t.tar_data)
            t.range_targets()
            trajs.append(t)
            # if you want to observe the trajectory generate procedure, enable the following codes.
            print('\rsrl train traj process:{:.2f}%'.format(i / simu_num * 100), end='')
        print('\rsrl train traj process done.')

        return trajs

    for v_name in ('empty20000x20000',):
        v_path = data_path + '\\rl\\srl\\vir'
        vscene = generator.load_scene(v_path + '\\' + v_name + '.json', simple_load=True)
        trajs = __obtain_srl_random_traj(0.02, vscene.max_size, 5000)
        for i, traj in enumerate(trajs):
            generator.save_trajectory(traj, v_path + '\\simu_trajs\\{}\\{}.json'.format(v_name, 'traj' + str(i)))
            print('\rsaving traj rationality: {:.2f}%'.format((i + 1) / len(trajs) * 100), end='')
        print()


def generate_swerc_live_pscene():
    w = 60  # original work use 10 x 10 phy spaces
    name = 'test11'
    contour = [
        [
            [0.5 * w, 0],
            [16.5 * w, 0],
            [16.5 * w, 12 * w],
            [14 * w, 12 * w],
            [14 * w, 8 * w],
            [3 * w, 8 * w],
            [3 * w, 12 * w],
            [0.5 * w, 12 * w],
        ],
        [
            [4 * w, 3 * w],
            [13 * w, 3 * w],
            [13 * w, 5 * w],
            [4 * w, 5 * w]
        ]

    ]

    phy_path = data_path + '\\phy'
    scene = DiscreteScene()
    scene.update_contours(name, contour)
    scene.update_segmentation()
    scene.update_grids_precompute_attr(True, True, True)
    generator.save_scene(scene, phy_path)
    generator.save_scene(scene, phy_path)


abbr_name = {
    'complex': 'cplx',
    'empty': 'ey',
    'mess': 'ms',
    'diff': 'df',
    'complex5000x5000': 'cplx50',
    'empty5000x5000': 'ey50',
    'empty2000x2000': 'ey20',
    'complex2000x2000': 'cplx20',

}

if __name__ == '__main__':
    # generate_srl_training_scene()
    # gc.collect()
    # generate_s2ot_training_scene()
    # gc.collect()
    # generate_src_training_pscenes()
    # gc.collect()
    # generate_src_training_vscenes()
    # gc.collect()
    # generate_src_vir_trajs()
    # gc.collect()
    # generate_srl_vir_trajs()
    # gc.collect()
    generate_swerc_live_pscene()
