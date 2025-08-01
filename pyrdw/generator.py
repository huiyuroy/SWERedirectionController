import numpy as np

import pyrdw.lib.math.geometry as geo
from pyrdw import *
from pyrdw.core.space.grid import Tiling
from pyrdw.core.space.roadmap import Node
from pyrdw.core.space.scene import Boundary
from pyrdw.core.space.scene import DiscreteScene
from pyrdw.core.space.trajectory import Trajectory
from pyrdw.util.file import load_json, save_json, save_grid_image, save_bin, load_bin

BOUND_ATTR_FORMAT = {
    "is_out_bound": bool,
    "points": list,
    "center": list,
    "barycenter": list,
    "cir_rect": list
}

NODE_ATTR_FORMAT = {
    "id": int,
    "pos": list,
    "loop_connect": list,
    "children_ids": list,
    "father_ids": list
}

TRI_ATTR_FORMAT = {
    "vertices": list,
    "barycenter": list,
    "in_circle": list,
    "out_edges": list,
    "in_edges": list
}
CONVEX_ATTR_FORMAT = {
    "vertices": list,
    "center": list,
    "barycenter": list,
    "cir_circle": list,
    "in_circle": list,
    "cir_rect": list,
    "out_edges": list,
    "in_edges": list
}

TILING_ATTR_FORMAT = {
    "id": int,
    "mat_loc": list,
    "center": list,
    "cross_bound": list,
    "rect": list,
    "h_width": float,
    "h_diag": float,
    "x_min": float,
    "x_max": float,
    "y_min": float,
    "y_max": float,
    "type": int
}

SCENE_ATTR_FORMAT = {
    "name": str,
    "bounds": list,
    "max_size": list,
    "out_bound_conv": dict,
    "out_conv_hull": dict,
    "scene_center": list,
    "tilings": list,
    "tilings_shape": list,
    "tiling_w": float,
    "tiling_x_offset": float,
    "tiling_y_offset": float,
    "tilings_data": list,
    "tilings_nei_ids": list,
    "tris": list,
    "tris_nei_ids": list,
    "conv_polys": list,
    "conv_nei_ids": list
}


def get_deep_size(obj, seen=None):
    """递归计算对象及其内容的总内存大小"""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_deep_size(v, seen) for v in obj.values())
        size += sum(get_deep_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_deep_size(i, seen) for i in obj)
    return size


def get_files(directory, extension):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.' + extension):
                filepath = os.path.join(root, filename)
                files.append(filepath)
    return files


def load_bound(bound_attr):
    bound = Boundary()
    bound.set_contour(bound_attr["is_out_bound"], bound_attr["points"])
    bound.center = np.array(bound_attr["center"])
    bound.barycenter = bound_attr["barycenter"]
    bound.cir_rect = bound_attr["cir_rect"]
    return bound


def load_tri(tri_attr):
    tri = geo.Triangle()
    tri.vertices = tuple(tri_attr["vertices"])
    tri.barycenter = tuple(tri_attr["barycenter"])
    tri.in_circle = tuple(tri_attr["in_circle"])
    tri.out_edges = tuple(tri_attr["out_edges"])
    tri.in_edges = tuple(tri_attr["in_edges"])
    return tri


def load_convex_poly(convex_attr):
    convex = geo.ConvexPoly()
    convex.vertices = tuple(convex_attr["vertices"])
    convex.poly_contour = Polygon(convex.vertices)
    convex.center = np.array(convex_attr["center"])
    convex.barycenter = tuple(convex_attr["barycenter"])
    convex.cir_circle = tuple(convex_attr["cir_circle"])
    convex.in_circle = tuple(convex_attr["in_circle"])
    convex.out_edges = tuple(convex_attr["out_edges"])
    convex.in_edges = tuple(convex_attr["in_edges"])
    return convex


def load_grid_base(scene, tiling_attr) -> Tiling:
    tiling = Tiling()
    tiling.corr_scene = scene
    tiling.id = tiling_attr["i"]
    tiling.mat_loc = tuple(tiling_attr["m_i"])
    tiling.type = tiling_attr["t"]
    tiling.cross_bound = np.array(tiling_attr["cb"])
    tiling.cross_area = [np.array(ca, dtype=np.float32) for ca in tiling_attr["ca"]]
    tiling.corr_conv_ids = tuple(tiling_attr["cid"])
    tiling.corr_conv_cin = tiling_attr["cin"]
    return tiling


def load_grid_visibility(tiling: Tiling, tiling_attr) -> Tiling:
    tiling.vis_tri = []
    for tri_attr in tiling_attr["d"]:
        vis_ori = np.array(tri_attr["o"])
        vis_tri = np.array(tri_attr["t"])
        if vis_tri.shape[0] > 0:
            n, _ = vis_tri.shape
            vis_tri = vis_tri.reshape(-1, 2, 2)
            ori = np.array([vis_ori.copy() for _ in range(n)]).reshape(n, 1, 2)
            vis_tri = np.concatenate((ori, vis_tri), axis=1)
        else:
            vis_tri = np.array([])
        tiling.vis_tri.append(vis_tri)
    tiling.vis_tri = tuple(tiling.vis_tri)
    return tiling


def load_tiling_extend(tiling, tiling_attr) -> Tiling:
    tiling.vis_grids = tuple(tiling_attr['v_g'])
    tiling.nearst_obs_pos = np.array(tiling_attr["n_obs_p"])
    tiling.sur_gids = tuple(tiling_attr['s_i'])
    tiling.sur_obs_gids = tuple(tiling_attr['s_obs_i'])
    tiling.sur_bound_gids = tuple(tiling_attr['s_obs_b_i'])
    tiling.sur_occu_safe = bool(tiling_attr['s_occu_safe'])

    return tiling


def load_contours(scene, data):
    bound_points = []
    in_bounds = []
    for bound in data['bounds']:
        points = bound['points']
        if bound['is_out_bound']:
            bound_points.append(points)
        else:
            in_bounds.append(points)
    bound_points.extend(in_bounds)
    scene.update_contours(data['name'], bound_points)


def load_segmentation(scene, data):
    scene.tris = []
    for tri_attr in data["tris"]:
        scene.tris.append(load_tri(tri_attr))
    scene.tris_nei_ids = data["tris_nei_ids"]
    scene.conv_polys = []
    for conv_attr in data["conv_polys"]:
        scene.conv_polys.append(load_convex_poly(conv_attr))
    scene.conv_nei_ids = data["conv_nei_ids"]
    scene.conv_area_priority = data["conv_area_priority"]
    scene.conv_collision_priority = data["conv_collision_priority"]


def load_grids(scene, base_data, vis_data, extend_data):
    scene.tilings = [None] * len(base_data["ts"])
    for tiling_attr in base_data["ts"]:
        tiling = load_grid_base(scene, tiling_attr)
        scene.tilings[tiling.id] = tiling
    if vis_data is not None:
        for tiling_attr in vis_data["ts"]:
            tiling = scene.tilings[tiling_attr["i"]]
            load_grid_visibility(tiling, tiling_attr)
    if extend_data is not None:
        for tiling_attr in extend_data["ts"]:
            tiling = scene.tilings[tiling_attr["i"]]
            load_tiling_extend(tiling, tiling_attr)
    scene.tilings = tuple(scene.tilings)
    scene.tilings_shape = tuple(base_data["tm_shape"])
    scene.tiling_w = base_data["tw"]
    scene.tiling_w_inv = 1 / scene.tiling_w
    scene.tiling_offset = np.array(base_data["toff"])


def load_roadmap(scene, data):
    def load_node(node_attr):
        n = Node()
        n.id = node_attr['id']
        n.pos = node_attr['pos']
        n.rela_loop_id = node_attr['loop_connect']
        n.child_ids = node_attr['children_ids']
        n.father_ids = node_attr['father_ids']
        return n

    nodes = list(map(load_node, data['data']))

    for node in nodes:
        if node.rela_loop_id != -1:
            for temp_node in nodes:
                if temp_node.id == node.rela_loop_id:
                    node.rela_loop_node = temp_node
                    break
        node.child_nodes = []
        for cid in node.child_ids:
            for temp_node in nodes:
                if temp_node.id == cid:
                    node.child_nodes.append(temp_node)
                    break
        node.father_nodes = []
        for fid in node.father_ids:
            for temp_node in nodes:
                if temp_node.id == fid:
                    node.father_nodes.append(temp_node)
                    break
    scene.nodes = nodes
    scene.update_roadmap()


def load_scene(tar_path,
               simple_load=False,
               load_road=False,
               load_vis=True,
               load_extend=True,
               scene_class=DiscreteScene):
    scene = scene_class()
    scene_dir, scene_name = os.path.split(tar_path)
    s_name = scene_name.split(".")[0]
    contour_data = load_json(tar_path)
    load_contours(scene, contour_data)
    if simple_load:
        return scene
    if load_road:
        road_data = load_json(scene_dir + '\\roadmap\\{}_rd.json'.format(s_name))
        load_roadmap(scene, road_data)
    segment_data = load_json(scene_dir + '\\segment\\{}_seg.json'.format(s_name))
    load_segmentation(scene, segment_data)

    grid_base = load_bin(scene_dir + '\\grid\\{}_base.bin'.format(s_name))
    grid_vis = None
    if load_vis:
        grid_vis = load_bin(scene_dir + '\\grid\\{}_visibility.bin'.format(s_name))
    grid_extend = None
    if load_extend:
        grid_extend = load_bin(scene_dir + '\\grid\\{}_extend.bin'.format(s_name))
    load_grids(scene, grid_base, grid_vis, grid_extend)

    return scene


def load_scene_base(tar_path, scene_class=DiscreteScene):
    scene = scene_class()
    scene_dir, scene_name = os.path.split(tar_path)
    s_name = scene_name.split(".")[0]
    contour_data = load_json(tar_path)
    load_contours(scene, contour_data)
    segment_data = load_json(scene_dir + '\\segment\\{}_seg.json'.format(s_name))
    load_segmentation(scene, segment_data)
    return scene


def load_scene_contour(tar_path, scene_class=DiscreteScene):
    scene = scene_class()
    scene_dir, scene_name = os.path.split(tar_path)
    s_name = scene_name.split(".")[0]
    contour_data = load_json(tar_path)
    load_contours(scene, contour_data)

    return scene


def load_trajectories(tar_path, scene_name):
    all_traj_files = get_files(tar_path + '\\simu_trajs\\{}'.format(scene_name), 'json')
    trajs = []
    total_traj_num = len(all_traj_files)
    for t_id, t in enumerate(map(load_trajectory, all_traj_files)):
        t.id = t_id
        trajs.append(t)
        print(f'\r{scene_name} loading trajs: {t_id / total_traj_num * 100:.2f}%', end='')
    print()

    return tuple(trajs)


def load_trajectory(tar_path):
    traj_data = load_json(tar_path)
    traj_type = traj_data['type']
    traj_tars = traj_data['targets']
    t = Trajectory()
    t.id = 0
    t.type = traj_type
    t.tar_data = tuple(traj_tars)
    t.tar_num = len(traj_tars)
    t.end_idx = t.tar_num - 1
    return t


def save_bound(bound):
    return {"is_out_bound": bound.is_out_bound,
            "points": np.array(np.around(bound.points, decimals=2), dtype='float').tolist(),
            "center": np.array(np.around(bound.center, decimals=2), dtype='float').tolist(),
            "barycenter": np.array(np.around(bound.barycenter, decimals=2), dtype='float').tolist(),
            "cir_rect": np.array(np.around(bound.cir_rect, decimals=2), dtype='float').tolist()}


def save_tri(tri):
    return {"vertices": np.array(np.around(tri.vertices, decimals=2), dtype='float').tolist(),
            "barycenter": np.array(np.around(tri.barycenter, decimals=2), dtype='float').tolist(),
            "in_circle": [np.array(np.around(tri.in_circle[0], decimals=2), dtype='float').tolist(),
                          float(np.around(tri.in_circle[1], decimals=2))],
            "out_edges": np.array(tri.out_edges).copy().tolist(),
            "in_edges": np.array(tri.in_edges).copy().tolist()}


def save_convex_poly(convex):
    return {"vertices": np.array(np.around(convex.vertices, decimals=2), dtype='float').tolist(),
            "center": np.array(np.around(convex.center, decimals=2), dtype='float').tolist(),
            "barycenter": np.array(np.around(convex.barycenter, decimals=2), dtype='float').tolist(),
            "cir_circle": [np.array(np.around(convex.cir_circle[0], decimals=2), dtype='float').tolist(),
                           float(np.around(convex.cir_circle[1], decimals=2))],
            "in_circle": [np.array(np.around(convex.in_circle[0], decimals=2), dtype='float').tolist(),
                          float(np.around(convex.in_circle[1], decimals=2))],
            "cir_rect": np.array(convex.cir_rect).copy().tolist(),
            "out_edges": np.array(convex.out_edges).copy().tolist(),
            "in_edges": np.array(convex.in_edges).copy().tolist()}


def save_contours(scene):
    return {'name': scene.name,
            'bounds': list(map(save_bound, scene.bounds)),
            'max_size': np.array(np.around(scene.max_size, decimals=2), dtype='float').tolist(),
            "out_bound_conv": save_convex_poly(scene.out_bound_conv),
            "out_conv_hull": save_convex_poly(scene.out_conv_hull),
            "scene_center": np.array(np.around(scene.scene_center, decimals=2), dtype='float').tolist()}


def save_segmentation(scene):
    return {"tris": list(map(save_tri, scene.tris)),
            "tris_nei_ids": pickle.loads(pickle.dumps(scene.tris_nei_ids)),
            "conv_polys": list(map(save_convex_poly, scene.conv_polys)),
            "conv_nei_ids": pickle.loads(pickle.dumps(scene.conv_nei_ids)),
            "conv_area_priority": np.array(np.around(scene.conv_area_priority, decimals=4), dtype='float').tolist(),
            "conv_collision_priority": np.array(np.around(scene.conv_collision_priority, decimals=4),
                                                dtype='float').tolist()}


def save_grids(scene):
    base_info = {"ts": tuple(map(save_grid_base, scene.tilings)),
                 "tm_shape": np.array(scene.tilings_shape).tolist(),
                 "tw": scene.tiling_w,
                 "toff": scene.tiling_offset.tolist()}
    vis_info = {"ts": tuple(map(save_grid_vis, scene.tilings))}
    extend_info = {"ts": tuple(map(save_grid_extend, scene.tilings))}
    return base_info, vis_info, extend_info


def save_grid_base(tiling: Tiling):
    return {"i": tiling.id,
            "m_i": tiling.mat_loc,
            "t": tiling.type,
            "cb": np.array(np.around(tiling.cross_bound, decimals=2), dtype='float').tolist(),
            "ca": [np.around(area, decimals=2).tolist() for area in tiling.cross_area],
            "cid": np.array(tiling.corr_conv_ids).tolist(),
            "cin": tiling.corr_conv_cin}


def save_grid_vis(tiling: Tiling):
    all_data = {"i": tiling.id,
                "d": []}
    for tri in tiling.vis_tri:
        data = np.array(np.around(tri, decimals=2))
        if data.shape[0] > 0:
            all_data["d"].append({"o": data[0, 0].tolist(), "t": data[:, 1:3, :].reshape(-1, 4).tolist()})
        else:
            all_data["d"].append({"o": [], "t": []})

    return all_data


def save_grid_extend(tiling):
    return {"i": tiling.id,
            'v_g': tiling.vis_grids,
            # "n_obs_i": tiling.nearst_obs_gid,
            "n_obs_p": np.around(tiling.nearst_obs_pos, 2).tolist(),
            's_i': tiling.sur_gids,
            's_obs_i': tiling.sur_obs_gids,
            's_obs_b_i': tiling.sur_bound_gids,
            's_occu_safe': tiling.sur_occu_safe}


def save_roadmap(scene):
    def save_node(n):
        return {'id': n.id,
                'pos': np.array(n.pos).tolist(),
                'loop_connect': np.array(n.rela_loop_id).tolist(),
                'children_ids': np.array(n.child_ids).tolist(),
                'father_ids': np.array(n.father_ids).tolist()}

    return {'name': scene.name, 'data': list(map(save_node, scene.nodes))}


def save_scene(scene, tar_path=None):
    contour_data = save_contours(scene)
    segment_data = save_segmentation(scene)
    grid_base, grid_vis, grid_extend = save_grids(scene)
    road_attr = save_roadmap(scene)
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)
    save_json(road_attr, tar_path + '\\roadmap\\{}_rd.json'.format(scene.name))
    save_json(contour_data, tar_path + '\\{}.json'.format(scene.name))
    save_json(segment_data, tar_path + '\\segment\\{}_seg.json'.format(scene.name))
    save_bin(grid_base, tar_path + '\\grid\\{}_base.bin'.format(scene.name))
    save_bin(grid_vis, tar_path + '\\grid\\{}_visibility.bin'.format(scene.name))
    save_bin(grid_extend, tar_path + '\\grid\\{}_extend.bin'.format(scene.name))

    if len(scene.tilings) > 0:
        data = np.zeros((scene.tilings_shape[1], scene.tilings_shape[0]))
        for t in scene.tilings:
            i, j = t.mat_loc
            data[i, j] = t.type
        save_grid_image(data * 255, tar_path + '\\{}.bmp'.format(scene.name))
    print('save {} done'.format(scene.name))


def save_trajectory(traj: Trajectory, tar_path):
    t_data = {'id': traj.id,
              'type': traj.type,
              'targets': np.around(traj.tar_data, decimals=2).tolist()}

    save_json(t_data, tar_path)
