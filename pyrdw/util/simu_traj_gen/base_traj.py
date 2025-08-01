import numpy as np

from pyrdw import *
import pyrdw.lib.math.algebra as alg
import pyrdw.lib.math.geometry as geo

from pyrdw.core.space.scene import DiscreteScene
from pyrdw.core.space.trajectory import Trajectory
from pyrdw.generator import load_scene, load_trajectories

PI = np.pi
RAD2DEG = 180 / np.pi
DEG2RAD = np.pi / 180


def gen_simu_road_trajs(scene: DiscreteScene,
                        patch_repeat,
                        prox=False,
                        dis_ang=5,
                        cut_off=100) -> List[Trajectory]:
    """
    生成用于训练的行走路线集，基本策略是将每个patch作为起始，尽量生成指定数量且不重复的路径轨迹。如果指定数量过大，生成的重复路径过多时可能
    导致算法反复尝试生成预期数量，解决策略是设置最大尝试次数，超出最大尝试次数直接转换到下一个patch开始。最终生成轨迹数量可能无法达到预期数量.

    Args:
        scene:
        patch_repeat: 每个patch为起始位置时，生成的轨迹数量
        prox: 是否对路径目标位置施加扰动（生成结果的路径会在一些路径点上与原始路径间存在扰动）,每次在原始路径两个target之间扰动一下，让原始
        方向随机向左或向右偏离10度之间（左右各5度）
        dis_ang: 扰动角度，默认是5度
        cut_off: 截断次数（100），当某个patch上轨迹生成尝试次数超过时则直接切换下一个patch

    Returns:
        trajs: 轨迹列表，包含n个Trajectory对象
    """

    patches = scene.patches
    patches_mat = scene.patches_mat
    simu_trajectories = []
    patch_record = []
    if len(patches_mat) == 1:
        for i in range(patch_repeat):
            patch_record.append([0])
    else:
        for start_patch_index in range(len(patches_mat)):
            current_recorded_patch_lists = []
            i = 0
            last_i = 0
            attempt_times = 0
            while i < patch_repeat and attempt_times < cut_off:
                if last_i == i:
                    attempt_times += 1
                current_patch_index = start_patch_index
                patch_walked_list = [current_patch_index]
                connected_patch_indexes = scene.find_nei_patches(current_patch_index)
                index = random.randint(0, len(connected_patch_indexes) - 1)
                current_patch_index = connected_patch_indexes[index]
                while current_patch_index != start_patch_index:
                    patch_walked_list.append(current_patch_index)
                    connected_patch_indexes = scene.find_nei_patches(current_patch_index)
                    index = random.randint(0, len(connected_patch_indexes) - 1)
                    current_patch_index = connected_patch_indexes[index]
                patch_walked_list.append(current_patch_index)
                can_add = 1
                for recorded_list in patch_record:
                    if operator.eq(recorded_list, patch_walked_list):
                        can_add = 0
                        break
                if can_add:
                    patch_record.append(patch_walked_list)
                    current_recorded_patch_lists.append(patch_walked_list)
                    i += 1
                    attempt_times = 0
                last_i = i
    if prox:
        print("generate prox road trajectories")
    else:
        print("generate abs road trajectories")

    patch_nums = len(patch_record)
    for p_id in range(patch_nums):
        patch_list = patch_record[p_id]
        i = 0
        list_len = len(patch_list)
        start_p = patches[patch_list[0]]
        if list_len > 1:
            relation = start_p.judge_relation_with_other_patch(patch_list[1])
            start_node = start_p.end_node if relation == "s" else start_p.start_node
        else:  # 如只有一条patch，暂时沿顺映射方向
            start_node = start_p.start_node
        cur_node = start_node.pos
        targets_list = []
        while i < list_len:
            cur_patch = patches[patch_list[i]]
            cur_nodes = cur_patch.nodes
            cur_p_s_node = cur_patch.start_node.pos
            cur_p_e_node = cur_patch.end_node.pos
            if geo.chk_p_same(cur_node, cur_p_s_node):  # 顺cur的顺序
                c_idx, c_step = 0, 1
                cur_node = cur_p_e_node
                i += 1
            elif geo.chk_p_same(cur_node, cur_p_e_node):
                c_idx, c_step = - 1, -1
                cur_node = cur_p_s_node
                i += 1
            else:
                i -= 1
                continue
            sign = 1
            pre_pos = cur_nodes[c_idx].pos
            while abs(c_idx) < len(cur_nodes):
                cur_pos = cur_nodes[c_idx].pos
                move_v = (np.array(cur_pos) - np.array(pre_pos)).tolist()
                if prox:
                    move_v = geo.rot_vecs(move_v, sign * random.randint(0, dis_ang) * DEG2RAD).tolist()
                sign *= -1
                targets_list.append((round(pre_pos[0] + move_v[0], 3), round(pre_pos[1] + move_v[1], 3)))
                pre_pos = cur_pos
                c_idx += c_step

        while True:
            find_repeat = False
            for k in range(len(targets_list) - 1):
                cur_t = targets_list[k]
                nxt_t = targets_list[k + 1]
                if geo.chk_p_same(cur_t[0:2], nxt_t[0:2]):
                    targets_list.pop(k + 1)
                    find_repeat = True
                    break
            if not find_repeat:
                break
        line_info = ""
        for k in range(len(targets_list)):
            line_info = "\rtraj id {} - steps {}/{}".format(p_id, k, len(targets_list)) + ""
            print(line_info, end="")
        verified, error_info = verify_single_traj(scene, line_info, targets_list)
        if verified:
            t = Trajectory()
            t.type = ('absolute' if not prox else 'approximate') + ' roadmap'
            t.tar_data = targets_list
            t.tar_num = len(targets_list)
            simu_trajectories.append(t)
        else:
            print('\nerror in building traj:' + str(error_info), end="")

    return simu_trajectories


def gen_simu_abs_rand_trajs(scene: DiscreteScene,
                            rand_repeat,
                            walk_range=None,
                            rot_range=None,
                            min_targets=120,
                            max_targets=1200) -> List[Trajectory]:
    """
    计算完全随机的行走轨迹，以秒为单位计算轨迹目标，每次1/3的概率进行转向。算法基于场景多边形区域分割，从每个分割区域选择rand_repeat个随机起点
    生成路径。

    Args:
        scene:
        rand_repeat: 遍历场景多边形剖分的每个多边形作为随机路径起点的种子，每个多边形区域重复生成的路径数量
        walk_range: 行走速度范围，默认0.2m/s - 1m/s
        rot_range: 旋转速度范围，默认 0 - pi in rad
        min_targets: 最少行走时间内，默认100s
        max_targets: 最大行走时间，默认500s

    Returns:
        trajs: 轨迹列表，包含n个Trajectory对象
    """

    conv_polys = scene.conv_polys
    simu_trajectories = []
    if rot_range is None:
        rot_range = [0, PI]
    if walk_range is None:
        walk_range = [20, 100]

    print("generate {} abs rand trajectories".format(len(conv_polys)*rand_repeat))

    rot_prob = 0.25  # 选择转向的概率是1/4
    act_attempt_times = 20  # 行走或者转向行为最多尝试20次，不行就直接切换动作状态
    min_vel, max_vel = walk_range
    min_rot, max_rot = rot_range
    total_epi = 0
    for conv in conv_polys:
        out_cir_center, out_cir_radius = conv.cir_circle
        min_x, max_x = out_cir_center[0] - out_cir_radius, out_cir_center[0] + out_cir_radius
        min_y, max_y = out_cir_center[1] - out_cir_radius, out_cir_center[1] + out_cir_radius
        repeat_epi = 0
        while repeat_epi < rand_repeat:
            s_x = np.random.uniform(min_x, max_x)
            s_y = np.random.uniform(min_y, max_y)
            if scene.poly_contour_safe.covers(Point(s_x, s_y)):
                cur_pos = [s_x, s_y]
                cur_vel = 100  # 初始速度定为 1m/s
                cur_fwd = geo.rot_vecs([0, 1], np.random.choice([1, -1]) * np.random.uniform() * PI).tolist()
                time_travel = np.random.uniform() * (max_targets - min_targets) + min_targets
                time_travel = math.ceil(time_travel)
                tars = []
                cur_step = 0
                force_rot = False
                force_mov = False
                tars.append(np.array(cur_pos))
                line_info = ""
                while cur_step <= time_travel:
                    if force_rot:
                        act = 'rot'
                    elif force_mov:
                        act = 'mov'
                    else:
                        state_prob = np.random.uniform()
                        act = 'rot' if state_prob <= rot_prob else 'mov'
                    if 'rot' in act:  #
                        force_rot = False
                        proc_rot = False
                        attempt_epi_id = 0
                        while not proc_rot:
                            temp_dir = np.random.choice([1, -1])
                            temp_vel = cur_vel
                            if attempt_epi_id < act_attempt_times:
                                rot_deg_prob = alg.clamp(np.random.normal(loc=0, scale=0.5), -1, 1)
                                temp_rot = (rot_deg_prob + 1) * 0.5 * (max_rot - min_rot) + min_rot
                            else:
                                temp_rot = np.random.uniform() * PI
                                temp_vel = min_vel
                            temp_fwd = geo.norm_vec(geo.rot_vecs(cur_fwd, temp_rot * temp_dir))
                            next_pos = np.array(cur_pos) + temp_fwd * temp_vel
                            if scene.poly_contour_safe.covers(LineString((cur_pos, next_pos))):
                                proc_rot = True
                                force_mov = True
                                cur_fwd = temp_fwd
                                cur_vel = 0
                            else:
                                attempt_epi_id += 1
                    else:
                        force_mov = False
                        proc_mov = False
                        attempt_epi_id = 0
                        while not proc_mov and attempt_epi_id < act_attempt_times:
                            mov_vel_prob = alg.clamp(np.random.normal(loc=0, scale=0.5), -1, 1)
                            temp_vel = (mov_vel_prob + 1) * 0.5 * (max_vel - min_vel) + min_vel
                            temp_fwd = geo.norm_vec(cur_fwd) * temp_vel
                            next_pos = np.array(cur_pos) + np.array(temp_fwd)
                            if scene.poly_contour_safe.covers(LineString((cur_pos, next_pos))):
                                line_info = "\rtraj {} - steps {}/{}".format(total_epi, cur_step, time_travel) + ""
                                print(line_info, end="")
                                proc_mov = True
                                cur_step += 1
                                cur_pos = next_pos.tolist()
                                # if not scene.poly_contour_safe.covers(Point(cur_pos)):
                                #     raise Exception("abs rand wrong")
                                tars.append((cur_pos[0], cur_pos[1]))
                            else:
                                attempt_epi_id += 1
                        if attempt_epi_id >= act_attempt_times:
                            force_rot = True
                            cur_vel = 0
                verified, error_info = verify_single_traj(scene, line_info, tars)
                if verified:
                    t = Trajectory()
                    t.type = 'absolute random'
                    t.tar_data = tars
                    t.tar_num = len(tars)
                    simu_trajectories.append(t)
                repeat_epi += 1
                total_epi += 1
    return simu_trajectories


def gen_simu_tiling_rand_trajs(scene: DiscreteScene,
                               rand_repeat,
                               min_travel=120,
                               max_travel=1200):
    """
    计算基于离散网格的随机行走轨迹，以秒为单位计算轨迹目标，每次1/3的概率进行转向。算法基于场景多边形区域分割，从每个分割区域选择rand_repeat个随机起点
    生成路径。

    Args:
        scene:
        rand_repeat:
        min_travel:
        max_travel:

    Returns:

    """
    simu_trajectories = []
    tilings = scene.tilings
    tilings_nei_ids = scene.tilings_nei_ids
    conv_polys = scene.conv_polys

    print("generate {} tiling rand trajectories".format(len(conv_polys)*rand_repeat))
    total_epi = 0
    max_search_depth = 5
    cal_pos = lambda t_t: [
        np.random.uniform(t_t.center[0] - scene.tiling_h_width, t_t.center[0] + scene.tiling_h_width),
        np.random.uniform(t_t.center[1] - scene.tiling_h_width, t_t.center[1] + scene.tiling_h_width)]
    for conv in conv_polys:
        init_ts = []
        for t in tilings:
            if geo.chk_p_in_conv(t.center, conv.vertices, conv.in_circle) and t.type:
                init_ts.append(t.id)
        if len(init_ts) > 0:
            repeat_epi = 0
            while repeat_epi < rand_repeat:
                pot_ts = init_ts
                cur_t_id = np.random.choice(pot_ts)
                cur_tiling = tilings[cur_t_id]
                cur_pos = cal_pos(cur_tiling)
                tile_travel = math.ceil(np.random.uniform() * (max_travel - min_travel) + min_travel)
                tars = []
                cur_step = 0
                tars.append((cur_pos[0], cur_pos[1]))
                line_info = ""
                tars_id = []
                while cur_step <= tile_travel:
                    search_tar_done = False
                    while not search_tar_done:
                        pot_ts.clear()
                        search_depth_mat = [0] * len(tilings)
                        search_depth_mat[cur_t_id] = 0
                        search_deque = deque([])
                        search_deque.append(cur_t_id)
                        while len(search_deque) > 0:
                            temp_t_id = search_deque.popleft()
                            if search_depth_mat[temp_t_id] < max_search_depth:
                                pot_ts.append(temp_t_id)
                                cur_neis = np.array(tilings_nei_ids[temp_t_id]).copy().tolist()
                                for n_id in cur_neis:
                                    nei_t = tilings[n_id]
                                    if (n_id not in search_deque) and nei_t.type:
                                        search_depth_mat[n_id] = search_depth_mat[temp_t_id] + 1
                                        search_deque.append(n_id)
                        temp_tar_id = np.random.choice(pot_ts)
                        temp_pos = cal_pos(tilings[temp_tar_id])
                        if scene.poly_contour.covers(LineString((tars[-1][:2], temp_pos))):
                            search_tar_done = True
                            cur_t_id = temp_tar_id
                            cur_pos = temp_pos

                    tars_id.append(cur_t_id)
                    tars.append((cur_pos[0], cur_pos[1]))
                    line_info = "\rtraj {} - steps {}/{}".format(total_epi, cur_step, tile_travel) + ""
                    print(line_info, end="")
                    cur_step += 1
                verified, error_info = verify_single_traj(scene, line_info, tars)
                if verified:
                    t = Trajectory()
                    t.type = 'tiling random'
                    t.tar_data = tars
                    t.tar_num = len(tars)
                    simu_trajectories.append(t)
                    repeat_epi += 1
                    total_epi += 1
                else:
                    print('\nerror in building traj:' + str(error_info), end="")
    return simu_trajectories


def verify_single_traj(scene, pre_info, tars):
    min_traj_len = 0.2
    tars_num = len(tars)
    wrong_info = None
    for i in range(tars_num - 1):
        ts, te = tars[i][:], tars[i + 1][:]
        if not scene.poly_contour.covers(LineString((ts, te))):
            print(pre_info + " verifying failed", end="")
            return False, wrong_info
        else:
            print(pre_info + " verifying: {:.2f}%".format((i+1) / (tars_num - 1) * 100), end="")
            traj_vec = np.array(te) - np.array(ts)
            if alg.l2_norm(traj_vec) >= min_traj_len:
                split_num = math.ceil(alg.l2_norm(traj_vec) / min_traj_len)
                split_step = 1 / split_num
                tns = []
                for s in range(split_num):
                    tn = split_step * s * traj_vec + np.array(ts)
                    tns.append(tn.tolist())
                tns.append(te)
    return True, {}


def correct_trajs(scene_path, scene_name):
    scene: DiscreteScene = load_scene(scene_path + '\\{}.json'.format(scene_name), simple_load=True)
    trajs: Sequence[Trajectory] = load_trajectories(scene_path, scene_name)
    for traj in trajs:
        tars = pickle.loads(pickle.dumps(traj.tar_data))
        tars = deque(tars)
        new_tars = []

        while len(tars) > 1:
            s_tar = tars.popleft()
            e_tar = tars[0]
            new_tars.append(s_tar)

            if not scene.poly_contour.contains(LineString([s_tar, e_tar])):
                new_tars.append(s_tar)
