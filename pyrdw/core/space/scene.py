from pyrdw.core.space import *
from pyrdw.core.space.boundary import Boundary
from pyrdw.core.space.grid import Tiling
from pyrdw.core.space.roadmap import Patch
from pyrdw.core.space.visibility import Observer, SimpleRay, Ray

from pyrdw.lib.math.geometry import Triangle, ConvexPoly

RESET_SAFE_DIS = const_reset['reset_trigger_dis']


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


class Scene:
    def __init__(self):
        self.name = None
        self.scene_type = 'vir'
        self.bounds = []
        self.poly_contour: Polygon = Polygon()
        self.poly_contour_safe: Polygon = Polygon()  # only used for representing safe area of scene.
        self.max_size = [0, 0]  # w,h
        self.out_bound_conv = geo.ConvexPoly()  # 外包围盒凸包-矩形
        self.out_conv_hull = geo.ConvexPoly()  # 外边界凸包
        self.scene_center = np.array([0, 0])
        self.nodes = []
        self.patches = []
        self.patches_mat = []  # 记录patch彼此关系的matrix
        self.tris: Sequence[Triangle] = []
        self.tris_nei_ids = []
        self.conv_polys: Sequence[ConvexPoly] = []
        self.conv_nei_ids = []
        self.conv_area_priority = []
        self.conv_collision_priority = []
        self.conv_connection_priority = []

    def update_contours(self, name, contours_points):
        """

        Args:
            name:
            contours_points: 轮廓点集，必须是二维数组

        Returns:

        """
        self.name = name
        self.poly_contour = Polygon(shell=contours_points[0], holes=contours_points[1:])
        self.poly_contour_safe = self.poly_contour.buffer(distance=-RESET_SAFE_DIS)

        self.bounds = []
        for contour in contours_points:
            bound = Boundary()
            bound.is_out_bound = False
            bound.points = contour
            bound.clean_repeat()
            bound.points_num = len(bound.points)
            np_contour = np.array(bound.points).copy()
            bound.center = (np_contour.sum(axis=0) / bound.points_num).tolist()
            bound.barycenter = geo.calc_poly_barycenter(np_contour).tolist()
            bound.cir_rect = geo.calc_cir_rect(np_contour)[0].tolist()
            self.bounds.append(bound)
        out_bound = self.bounds[0]
        out_bound.is_out_bound = True
        out_bound_points = np.array(out_bound.points).copy().tolist()

        self.max_size = np.array(out_bound.cir_rect).max(axis=0)
        mx, my = self.max_size
        self.scene_center = np.array([mx * 0.5, my * 0.5])
        self.out_conv_hull = geo.ConvexPoly(geo.calc_convex_graham_scan(out_bound_points))
        self.out_bound_conv = geo.ConvexPoly([[0, 0], [mx, 0], [mx, my], [0, my]])
        print('{} contours: done'.format(self.name))

    def update_segmentation(self):
        """
            1. 耳分法生成三角剖分
            2. 基于三角剖分集进行区域生长，将三角形合并为多个凸多边形。每次生长都先选择面积最大的三角形作为生长中心，尽量将区域中最大的凸多边形生成出来
            注：边界轮廓需要按照特定顺序，外轮廓必须在列表首位

            Returns:

        """
        poly_bounds = [bound.points for bound in self.bounds]
        tris = geo.calc_poly_triangulation(poly_bounds)

        tris_num = len(tris)
        tris_nei_ids = [[] for _ in range(tris_num)]
        for i in range(tris_num):
            tar_tri = tris[i]
            for j in range(tris_num - 1):
                other_i = (j + i + 1) % tris_num
                other_tri = tris[other_i]
                if tar_tri.det_common_edge(other_tri) is not None:
                    tris_nei_ids[i].append(other_i)
        conv_polys = []
        total_area = 0
        total_perimeter = 0
        visit_mat = pickle.loads(pickle.dumps(tris_nei_ids))
        tris_ids = [i for i in range(len(tris))]
        grown_deque = deque()
        total_len = len(tris_ids)

        while len(tris_ids) > 0:
            start_tri_id = 0
            min_area = float('inf')
            max_area = 0
            for id in tris_ids:
                t_a = tris[id].poly_contour.area
                if t_a > max_area:
                    start_tri_id = id
                    max_area = t_a
            cur_poly = np.array(tris[start_tri_id].vertices).copy().tolist()
            cur_out_edges = []
            cur_in_edges = []
            for e in tris[start_tri_id].out_edges:
                v1_id, v2_id = e
                v1 = tris[start_tri_id].vertices[v1_id]
                v2 = tris[start_tri_id].vertices[v2_id]
                e = [np.array(v1).copy().tolist(), np.array(v2).copy().tolist()]
                cur_out_edges.append(e)
            for e in tris[start_tri_id].in_edges:
                v1_id, v2_id = e
                v1 = tris[start_tri_id].vertices[v1_id]
                v2 = tris[start_tri_id].vertices[v2_id]
                e = [np.array(v1).copy().tolist(), np.array(v2).copy().tolist()]
                cur_in_edges.append(e)
            tris_ids.remove(start_tri_id)  # 选定生长中心，则从三角形集中移除当前三角形
            grown_deque.clear()
            for n_id in visit_mat[start_tri_id]:
                if n_id in tris_ids:
                    grown_deque.append(n_id)
                    if start_tri_id in visit_mat[n_id]:
                        visit_mat[n_id].remove(start_tri_id)
            while len(grown_deque) > 0:
                cur_tri_id = grown_deque.popleft()
                cur_tri = tris[cur_tri_id].vertices
                find_con = False
                for nt_i in range(-1, 2):
                    tv1 = cur_tri[nt_i]
                    tv2 = cur_tri[nt_i + 1]
                    for p_i in range(-1, len(cur_poly) - 1):
                        pv1 = cur_poly[p_i]
                        pv2 = cur_poly[p_i + 1]
                        if geo.chk_edge_same([pv1, pv2], [tv1, tv2]):  # 当前多边形与目标三角形邻接
                            find_con = True
                            temp_poly = np.array(cur_poly).copy().tolist()
                            temp_poly.insert(p_i + 1, cur_tri[(nt_i + 2) % 3])
                            temp_poly, _ = geo.calc_adjust_poly_order(temp_poly, 1)
                            if geo.chk_poly_concavity(temp_poly):  # 能够成凸多边形
                                cur_poly = temp_poly
                                common_edge = [pv1, pv2]
                                for e in tris[cur_tri_id].out_edges:
                                    v1_id, v2_id = e
                                    v1 = cur_tri[v1_id]
                                    v2 = cur_tri[v2_id]
                                    e = [np.array(v1).copy().tolist(), np.array(v2).copy().tolist()]
                                    cur_out_edges.append(e)
                                for e in tris[cur_tri_id].in_edges:
                                    v1_id, v2_id = e
                                    v1 = cur_tri[v1_id]
                                    v2 = cur_tri[v2_id]
                                    e = [np.array(v1).copy().tolist(), np.array(v2).copy().tolist()]
                                    if not geo.chk_edge_same(e, common_edge):
                                        cur_in_edges.append(e)
                                for e in cur_in_edges:
                                    if geo.chk_edge_same(e, common_edge):
                                        cur_in_edges.remove(e)
                                tris_ids.remove(cur_tri_id)
                                nei_ids = visit_mat[cur_tri_id]
                                for n_id in nei_ids:
                                    visit_mat[n_id].remove(cur_tri_id)
                                    is_in_queue = False
                                    for q_id in grown_deque:
                                        if q_id == n_id:
                                            is_in_queue = True
                                    if not is_in_queue:
                                        grown_deque.append(n_id)
                            break
                    if find_con:
                        break

            conv_poly = geo.ConvexPoly(cur_poly)
            conv_poly.out_edges_perimeter = 0
            conv_poly.out_edges = []
            for e in cur_out_edges:
                v1_id = conv_poly.find_vertex_idx(e[0])
                v2_id = conv_poly.find_vertex_idx(e[1])
                conv_poly.out_edges.append([v1_id, v2_id])
                conv_poly.out_edges_perimeter += alg.l2_norm(
                    np.array(conv_poly.vertices[v1_id]) - np.array(conv_poly.vertices[v2_id]))
            conv_poly.in_edges = []
            for e in cur_in_edges:
                v1_id = conv_poly.find_vertex_idx(e[0])
                v2_id = conv_poly.find_vertex_idx(e[1])
                conv_poly.in_edges.append([v1_id, v2_id])
            conv_poly.area = conv_poly.poly_contour.area
            conv_polys.append(conv_poly)
            print('\r{} segmentation:{:.2f}%'.format(self.name, (total_len - len(tris_ids)) / total_len * 100), end='')
        print()
        poly_num = len(conv_polys)
        conv_nei_ids = [[] for _ in range(poly_num)]
        conv_area_priority = [0] * poly_num
        conv_collision_priority = [0] * poly_num
        conv_connection_priority = [0] * poly_num
        for i in range(poly_num):
            tar_p = conv_polys[i]
            total_area += tar_p.area
            total_perimeter += tar_p.out_edges_perimeter

            for j in range(poly_num - 1):
                other_i = (j + i + 1) % poly_num
                other_p = conv_polys[other_i]
                if tar_p.det_common_edge(other_p) is not None:
                    conv_nei_ids[i].append(other_i)
        for i in range(poly_num):
            tar_p = conv_polys[i]
            conv_area_priority[i] = tar_p.area / total_area
            conv_collision_priority[i] = 1 - tar_p.out_edges_perimeter / total_perimeter
            conv_connection_priority[i] = len(conv_nei_ids[i]) / (2 * poly_num)

        self.tris = tris
        self.tris_nei_ids = tris_nei_ids
        self.conv_polys = conv_polys
        self.conv_nei_ids = conv_nei_ids
        self.conv_area_priority = conv_area_priority
        self.conv_collision_priority = conv_collision_priority
        self.conv_connection_priority = conv_connection_priority

    def update_roadmap(self):
        """

        Args:
            nodes: Node对象数组

        Returns:

        """

        # with Progress() as progress:
        #     task = progress.add_task('roadmap generate:', total=100)
        iter_node = None
        for node in self.nodes:
            if len(node.father_nodes) == 0:
                iter_node = node
                break
        patches = []
        index = 0
        while True:
            patch = Patch()
            patches.append(patch)
            patch.id = index
            patch.start_node = iter_node
            random_select_iteration = False
            multi_child_nodes_appear = 0
            while True:
                patch.nodes.append(iter_node)
                iter_node.visited = True
                if len(iter_node.child_nodes) == 1 and iter_node.rela_loop_node is None:
                    iter_node = iter_node.child_nodes[0]
                elif len(iter_node.child_nodes) == 0:
                    patch.end_node = iter_node
                    random_select_iteration = True
                    break
                elif len(iter_node.child_nodes) > 1:
                    multi_child_nodes_appear = multi_child_nodes_appear + 1
                    if multi_child_nodes_appear == 1:
                        for child_node in iter_node.child_nodes:
                            if not child_node.visited:
                                iter_node = child_node
                                break
                    else:
                        patch.end_node = iter_node
                        break
                else:
                    multi_child_nodes_appear = multi_child_nodes_appear + 1
                    if multi_child_nodes_appear == 1:
                        for child_node in iter_node.child_nodes:
                            if not child_node.visited:
                                iter_node = child_node
                                break
                    else:
                        patch.end_node = iter_node
                        break
            patch.nodes_num = len(patch.nodes)
            if random_select_iteration:
                can_find = False
                for node in self.nodes:
                    if len(node.child_nodes) > 1:
                        for child_node in node.child_nodes:
                            if not child_node.visited:
                                can_find = True
                                iter_node = node
                                break
                    if can_find:
                        break
                if not can_find:
                    break
            index = index + 1
        for patch in patches:
            start_node = patch.start_node
            end_node = patch.end_node
            for other_patch in patches:
                if other_patch.id != patch.id:
                    other_start = other_patch.start_node
                    other_end = other_patch.end_node
                    if other_end.id == start_node.id or other_start.id == start_node.id:
                        patch.start_nb_patches_id.append(other_patch.id)
                    if other_start.id == end_node.id or other_end.id == end_node.id:
                        patch.end_nb_patches_id.append(other_patch.id)
                    if start_node.rela_loop_node is not None:
                        if other_end.id == start_node.rela_loop_node.id:
                            patch.start_nb_patches_id.append(other_patch.id)
                            other_patch.end_nb_patches_id.append(patch.id)
                        if other_start.id == start_node.rela_loop_node.id:
                            patch.start_nb_patches_id.append(other_patch.id)
                            other_patch.start_nb_patches_id.append(patch.id)
                        if other_end.rela_loop_node is not None:
                            if other_end.rela_loop_node.id == start_node.rela_loop_node.id:
                                patch.start_nb_patches_id.append(other_patch.id)
                                other_patch.end_nb_patches_id.append(patch.id)
                        if other_start.rela_loop_node is not None:
                            if other_start.rela_loop_node.id == start_node.rela_loop_node.id:
                                patch.start_nb_patches_id.append(other_patch.id)
                                other_patch.start_nb_patches_id.append(patch.id)
                    if end_node.rela_loop_node is not None:
                        if other_start.id == end_node.rela_loop_node.id:
                            patch.end_nb_patches_id.append(other_patch.id)
                            other_patch.start_nb_patches_id.append(patch.id)
                        if other_end.id == end_node.rela_loop_node.id:
                            patch.end_nb_patches_id.append(other_patch.id)
                            other_patch.end_nb_patches_id.append(patch.id)
                        if other_start.rela_loop_node is not None:
                            if other_start.rela_loop_node.id == end_node.rela_loop_node.id:
                                patch.end_nb_patches_id.append(other_patch.id)
                                other_patch.start_nb_patches_id.append(patch.id)
                        if other_end.rela_loop_node is not None:
                            if other_end.rela_loop_node.id == end_node.rela_loop_node.id:
                                patch.end_nb_patches_id.append(other_patch.id)
                                other_patch.end_nb_patches_id.append(patch.id)
        for patch in patches:
            list1 = patch.start_nb_patches_id
            list2 = list(set(list1))
            patch.start_nb_patches_id = list2
            for p_id in patch.start_nb_patches_id:
                for other_patch in patches:
                    if other_patch.id == p_id:
                        patch.start_nb_patches.append(other_patch)
                        if patch.start_node.id == other_patch.start_node.id:
                            patch.start_nb_patches_order.append(True)
                        elif patch.start_node.id == other_patch.end_node.id:
                            patch.start_nb_patches_order.append(False)
                        elif patch.start_node.rela_loop_node is not None and \
                                patch.start_node.rela_loop_node.id == other_patch.start_node.id:
                            patch.start_nb_patches_order.append(True)
                        elif patch.start_node.rela_loop_node is not None and \
                                patch.start_node.rela_loop_node.id == other_patch.end_node.id:
                            patch.start_nb_patches_order.append(False)
                        elif other_patch.start_node.rela_loop_node is not None and \
                                patch.start_node.id == other_patch.start_node.rela_loop_node.id:
                            patch.start_nb_patches_order.append(True)
                        elif other_patch.end_node.rela_loop_node is not None and \
                                patch.start_node.id == other_patch.end_node.rela_loop_node.id:
                            patch.start_nb_patches_order.append(False)
                        elif patch.start_node.rela_loop_node is not None and \
                                other_patch.start_node.rela_loop_node is not None and \
                                patch.start_node.rela_loop_node.id == other_patch.start_node.rela_loop_node.id:
                            patch.start_nb_patches_order.append(True)
                        elif patch.start_node.rela_loop_node is not None and \
                                other_patch.end_node.rela_loop_node is not None and \
                                patch.start_node.rela_loop_node.id == other_patch.end_node.rela_loop_node.id:
                            patch.start_nb_patches_order.append(False)
                        break
            list1 = patch.end_nb_patches_id
            list2 = list(set(list1))
            patch.end_nb_patches_id = list2
            for p_id in patch.end_nb_patches_id:
                for other_patch in patches:
                    if other_patch.id == p_id:
                        patch.end_nb_patches.append(other_patch)
                        if patch.end_node.id == other_patch.start_node.id:
                            patch.end_nb_patches_order.append(False)
                        elif patch.end_node.id == other_patch.end_node.id:
                            patch.end_nb_patches_order.append(True)
                        elif patch.end_node.rela_loop_node is not None and \
                                patch.end_node.rela_loop_node.id == other_patch.start_node.id:
                            patch.end_nb_patches_order.append(False)
                        elif patch.end_node.rela_loop_node is not None and \
                                patch.end_node.rela_loop_node.id == other_patch.end_node.id:
                            patch.end_nb_patches_order.append(True)
                        elif other_patch.start_node.rela_loop_node is not None and \
                                patch.end_node.id == other_patch.start_node.rela_loop_node.id:
                            patch.end_nb_patches_order.append(False)
                        elif other_patch.end_node.rela_loop_node is not None and \
                                patch.end_node.id == other_patch.end_node.rela_loop_node.id:
                            patch.end_nb_patches_order.append(True)
                        elif patch.end_node.rela_loop_node is not None and \
                                other_patch.start_node.rela_loop_node is not None and \
                                patch.end_node.rela_loop_node.id == other_patch.start_node.rela_loop_node.id:
                            patch.end_nb_patches_order.append(False)
                        elif patch.end_node.rela_loop_node is not None and \
                                other_patch.end_node.rela_loop_node is not None and \
                                patch.end_node.rela_loop_node.id == other_patch.end_node.rela_loop_node.id:
                            patch.end_nb_patches_order.append(True)
                        break
        max_id = 0
        for patch in patches:
            if patch.id > max_id:
                max_id = patch.id
        max_id += 1
        patches_mat = [[] for _ in range(max_id)]
        for patch in patches:
            index = patch.id
            patches_mat[index] = [0 for _ in range(max_id)]
            for other_patch in patch.start_nb_patches:
                other_index = other_patch.id
                patches_mat[index][other_index] = 1
            for other_patch in patch.end_nb_patches:
                other_index = other_patch.id
                patches_mat[index][other_index] = 1

        self.patches = patches
        self.patches_mat = patches_mat
        print('roadmap generate: done')

    def check_triangulation(self):
        tris = self.tris
        done1, done2, done3 = False, True, True
        # 首先检查三角剖分集的面积是否与原边界面积吻合
        tris_area = 0
        for tri in tris:
            tris_area += tri.poly_contour.area
        poly_area = self.poly_contour.area
        # for bound in self.bounds:
        #     if bound.is_out_bound:
        #         poly_area += geo.calc_poly_area(np.array(bound.points))
        #     else:
        #         poly_area -= geo.calc_poly_area(np.array(bound.points))
        if abs(tris_area - poly_area) < EPS:
            done1 = True
        # 其次检查三角剖分集是否存在三角形交叉的情况
        tri_num = len(tris)
        for i in range(tri_num - 1):
            tri1 = tris[i]
            for j in range(i + 1, tri_num):
                tri2 = tris[j]
                intersect = tri1.poly_contour.intersection(tri2.poly_contour)
                if isinstance(intersect, Polygon):
                    done2 = False
                    break
        # 最后检查三角剖分集是否存在非三角形的情况（轮廓三个点共线）
        for tri in tris:
            if not geo.chk_is_triangle(tri.vertices):
                done3 = False
        if done1 and done2 and done3:
            return True
        else:
            return False

    def obtain_tri_neis(self, tri_id):
        if tri_id > len(self.tris):
            raise Exception('wrong triangle index, max id: {}, input: {}'.format(len(self.tris) - 1, tri_id))
        else:
            return self.tris_nei_ids[tri_id]

    def obtain_conv_neis(self, conv_id):
        if conv_id > len(self.conv_polys):
            raise Exception('wrong convex index, max id: {}, input: {}'.format(len(self.conv_polys) - 1, conv_id))
        else:
            return self.conv_nei_ids[conv_id]

    def find_nei_patches(self, patch_index):
        list_id = []
        for i in range(len(self.patches_mat[patch_index])):
            if self.patches_mat[patch_index][i]:
                list_id.append(i)
        return list_id

    def clone(self):
        c_scene = Scene()
        for bound in self.bounds:
            c_scene.bounds.append(bound.clone())
        c_scene.max_size = np.array(self.max_size).copy().tolist()
        c_scene.scene_center = self.scene_center.copy()

        # tiling、patch、node clone还未完成
        # for tiling in self.tilings:
        #     c_scene.tilings.append(tiling.clone())
        # c_scene.tilings_shape = copy.deepcopy(self.tilings_shape)
        # c_scene.tiling_w = self.tiling_w

        for tri in self.tris:
            c_scene.tris.append(tri.clone())
        c_scene.tris_nei_ids = np.array(self.tris_nei_ids).copy().tolist()
        for conv in self.conv_polys:
            c_scene.conv_polys.append(conv.clone())
        c_scene.conv_nei_ids = np.array(self.conv_nei_ids).copy().tolist()

        return c_scene

    def clear(self):
        if self.bounds is not None:
            self.bounds.clear()
            self.max_size = [0, 0]
            self.scene_center = np.array([0, 0])
        if self.patches is not None:
            self.patches.clear()
        if self.nodes is not None:
            self.nodes.clear()
        if self.tris is not None:
            self.tris.clear()
            self.tris_nei_ids.clear()
        if self.conv_polys is not None:
            self.conv_polys.clear()
            self.conv_nei_ids.clear()


class DiscreteScene(Scene):

    def __init__(self):
        super().__init__()
        self.tiling_h_width: float = 0
        self.tiling_h_diag = 0
        self.tilings: Tuple[Tiling] = ()
        self.tilings_shape: Tuple = ()  # tilings 行列分布 [w,h]
        self.tiling_w: float = GRID_WIDTH  # tiling块的宽度大小，以m计算
        self.tiling_w_inv: float = 1 / GRID_WIDTH
        self.tiling_offset: np.ndarray = np.array([0, 0])
        self.tilings_data: np.ndarray = np.array([])
        self.tilings_weights: np.ndarray = np.array([])
        self.tilings_weights_grad: np.ndarray = np.array([])  # 梯度方向为weight下降方向，要用需要取反
        self.tilings_nei_ids = []
        self.tilings_walkable = []
        self.tilings_visitable = []
        self.tilings_rot_occupancy_num = 360
        self.tilings_rot_occupancy_grids = None
        self.vispoly_observer: Observer = None

    # belows are offline calculations, used for program boosting.
    def __grids_base_attr(self):
        """
        generate base attributes of scene grids, including:
            - tiling width
            - tiling offset to the scene center
            - tiling number shape: (w, h) w: horizontal number of tilings, h: vertical number of tilings
            - all tilings, each tiling contain: id, idx_loc, center, xy min and max, nei_ids


        Returns:

        """
        m_wd, m_wh = self.max_size
        w, h = math.ceil(m_wd / self.tiling_w), math.ceil(m_wh / self.tiling_w)
        self.tilings_shape = (w, h)
        self.tiling_h_width = self.tiling_w * 0.5
        self.tiling_h_diag = self.tiling_h_width * (2 ** 0.5)
        self.tiling_offset = np.array([(m_wd - w * self.tiling_w) * 0.5, (m_wh - h * self.tiling_w) * 0.5])
        self.tilings = [None] * (w * h)
        nei_offset = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        for i in range(h):
            for j in range(w):
                tiling: Tiling = Tiling()
                tiling.id = i * w + j
                tiling.mat_loc = (i, j)
                tc = np.array([(j + (1 - w) * 0.5) * self.tiling_w + 0.5 * m_wd,
                               (i + (1 - h) * 0.5) * self.tiling_w + 0.5 * m_wh])
                tiling.center = tc
                tiling.rect = np.array([(tc[0] - self.tiling_h_width, tc[1] - self.tiling_h_width),  # letf-bottom
                                        (tc[0] + self.tiling_h_width, tc[1] - self.tiling_h_width),  # right-bottom
                                        (tc[0] + self.tiling_h_width, tc[1] + self.tiling_h_width),  # right-top
                                        (tc[0] - self.tiling_h_width, tc[1] + self.tiling_h_width)])  # left-top
                tiling.poly_contour = Polygon(tiling.rect)

                nei_ids = []
                for off in nei_offset:
                    row, col = i + off[0], j + off[1]
                    if 0 <= row < h and 0 <= col < w:
                        nei_ids.append(row * w + col)
                tiling.nei_ids = tuple(nei_ids)
                self.tilings[tiling.id] = tiling
                print('\r{} grids generating: {:.2f}%'.format(self.name, (tiling.id + 1) / (w * h) * 100), end='')
        print()

        for t_idx, tiling in enumerate(self.tilings):
            tiling.rela_patch = None
            tiling.type = 0
            tiling.corr_conv_ids = []
            tiling.corr_conv_cin = -1
            find_intersect = False
            if self.patches is not None and len(self.patches) > 0:
                for patch in self.patches:
                    for i in range(patch.nodes_num - 1):
                        line_s = patch.nodes[i]
                        line_e = patch.nodes[i + 1]
                        l_pos = [line_s.pos[0], line_s.pos[1], line_e.pos[0], line_e.pos[1]]
                        if geo.chk_line_rect_cross(l_pos, tiling.rect):
                            tiling.rela_patch = patch.id
                            find_intersect = True
                            break
                    if find_intersect:
                        break
            tiling.intersection_scene(self)
            if tiling.type or tiling.cross_bound.shape[0] > 0:  # tiling is within or partially within the scene
                for i in range(len(self.conv_polys)):
                    conv = self.conv_polys[i]
                    inter = conv.poly_contour.intersection(tiling.poly_contour)
                    if not inter.is_empty and isinstance(inter, Polygon):
                        tiling.corr_conv_ids.append(i)
                    if conv.poly_contour.contains(Point(tiling.center)):
                        tiling.corr_conv_cin = i
            print('\r{} grids base attr building: {:.2f}%'.format(self.name, (t_idx + 1) / len(self.tilings) * 100),
                  end='')
        print()

    def __grids_vis_build(self, enable_grid=False):
        def ensure_vis_ori(pot_area):
            t_poly = Polygon(shell=pot_area)
            ori = None
            if t_poly.contains(Point(tiling.center)):
                ori = tiling.center
            else:
                p_min = pot_area.min(axis=0)
                p_max = pot_area.max(axis=0)
                find_ori = False
                while not find_ori:
                    rand_p = np.array(
                        [np.random.uniform(p_min[0], p_max[0]), np.random.uniform(p_min[1], p_max[1])])
                    p = Point(rand_p)
                    if self.poly_contour.contains(p) and t_poly.contains(p):
                        find_ori = True
                        ori = rand_p
            return ori

        self.enable_visibility()
        self.tilings_visitable = [1 if t.type or t.cross_bound.shape[0] > 0 else 0 for t in self.tilings]
        for idx, tiling in enumerate(self.tilings):
            tiling.vis_grids = []
            if tiling.type:
                v_tri, v_grid = self.update_visibility(tiling.center, grids_comp=enable_grid)
                tiling.vis_tri = (v_tri,)
                tiling.vis_grids = (v_grid,)
            elif tiling.cross_bound.shape[0] > 0:
                if len(tiling.cross_area) == 1:  # tiling 与场景仅有一个交叉区域
                    v_ori = ensure_vis_ori(tiling.cross_area[0])
                    v_tri, v_grid = self.update_visibility(v_ori, grids_comp=enable_grid)
                    tiling.vis_tri = (v_tri,)
                    tiling.vis_grids = (v_grid,)
                else:
                    vis_tri = []
                    vis_grid = []
                    for c_poly in tiling.cross_area:
                        v_ori = ensure_vis_ori(c_poly)
                        v_tri, v_grid = self.update_visibility(v_ori, grids_comp=enable_grid)
                        vis_tri.append(v_tri)
                        vis_grid.append(v_grid)
                    tiling.vis_tri = tuple(vis_tri)
                    tiling.vis_grids = tuple(vis_grid)
            else:
                tiling.vis_tri = ()

            print('\r{} grids vis building: {:.2f}%'.format(self.name, (idx + 1) / len(self.tilings) * 100), end='')
        print()

    def __grids_vis_occu_discrete(self, vis_radius=HUMAN_STEP):
        """

        Args:
            vis_radius:

        Returns:

        """

        w, h = self.tilings_shape
        for idx, tiling in enumerate(self.tilings):
            if tiling.type:  # be sure the tiling is within the scene
                min_dis = float('inf')
                closest_p = None
                for bound in self.bounds:
                    b_ps_num = len(bound.points)
                    for i in range(-1, b_ps_num - 1):
                        b_s, b_e = bound.points[i], bound.points[i + 1]
                        pro_p, _ = geo.calc_p_project_on_line(tiling.center, np.array([b_s, b_e]))
                        p_dis = alg.l2_norm(pro_p - tiling.center)
                        if p_dis < min_dis:
                            min_dis = p_dis
                            closest_p = pro_p
                tiling.nearst_obs_pos = closest_p
            else:  # tiling outside scene
                tiling.nearst_obs_pos = tiling.center

            r, c = tiling.mat_loc
            tiling.sur_gids = []
            tiling.sur_obs_gids = []
            tiling.sur_bound_gids = []
            radius_idx = math.ceil(vis_radius / self.tiling_w)
            sur1, sur2 = self.calc_tiling_diffusion((c, r), (0, 1), 360, radius_idx, 0)
            sur_tiling_ids = sur1 + sur2
            for sur_id_c, sur_id_r in sur_tiling_ids:
                sur_id = sur_id_r * w + sur_id_c
                sur_tiling = self.tilings[sur_id]
                tiling.sur_gids.append(sur_id)
                if not sur_tiling.type:
                    tiling.sur_obs_gids.append(sur_id)
                    if sur_tiling.cross_bound.shape[0] > 0:
                        tiling.sur_bound_gids.append(sur_id)
            tiling.sur_gids = tuple(tiling.sur_gids)
            tiling.sur_obs_gids = tuple(tiling.sur_obs_gids)
            tiling.sur_bound_gids = tuple(tiling.sur_bound_gids)
            for sur_id in tiling.sur_gids:
                sur_grid = self.tilings[sur_id]
                if not sur_grid.type:
                    tiling.sur_occu_safe = False
                    break

            print('\r{} grids occu relation building: {:.2f}%'.format(self.name, (idx + 1) / len(self.tilings) * 100),
                  end='')
        print()

    def update_grids_precompute_attr(self,
                                     enable_vis,
                                     enable_vis_grid,
                                     enable_discrete):
        """
        scene grids computation

        Args:

            enable_vis: 预计算场景可见性
            enable_vis_grid: only enable in phy scene
            enable_discrete: 预计算场景每个采样点的离散可见区域

        Returns:

        """
        self.__grids_base_attr()
        if enable_vis:
            self.__grids_vis_build(enable_vis_grid)
        if enable_discrete:
            self.__grids_vis_occu_discrete()

    def update_grids_runtime_attr(self):
        """
        calculate the runtime attributes of grids, used for supply offline setups

        Returns:

        """
        for bound in self.bounds:
            if bound.is_out_bound:
                bound.points, _ = geo.calc_adjust_poly_order(bound.points, order=0)  # 必须全部方向调整为顺时针
            else:
                bound.points, _ = geo.calc_adjust_poly_order(bound.points, order=1)  # 必须全部方向调整为逆时针

        m_wd, m_wh = self.max_size
        self.tiling_h_width = self.tiling_w * 0.5
        self.tiling_h_diag = self.tiling_h_width * (2 ** 0.5)
        self.tilings_data = []
        self.tilings_nei_ids = [[] for _ in range(len(self.tilings))]
        self.tilings_walkable = []
        self.tilings_visitable = np.zeros(len(self.tilings))
        w, h = self.tilings_shape
        nei_offset = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        for count, tiling in enumerate(self.tilings):
            i, j = tiling.mat_loc
            tc = np.array([(j + (1 - w) * 0.5) * self.tiling_w + 0.5 * m_wd,
                           (i + (1 - h) * 0.5) * self.tiling_w + 0.5 * m_wh])
            tiling.center = tc
            tiling.rect = np.array([(tc[0] - self.tiling_h_width, tc[1] - self.tiling_h_width),  # letf-bottom
                                    (tc[0] + self.tiling_h_width, tc[1] - self.tiling_h_width),  # right-bottom
                                    (tc[0] + self.tiling_h_width, tc[1] + self.tiling_h_width),  # right-top
                                    (tc[0] - self.tiling_h_width, tc[1] + self.tiling_h_width)])  # left-top
            tiling.poly_contour = Polygon(tiling.rect)
            nei_ids = []
            for off in nei_offset:
                row, col = i + off[0], j + off[1]
                if 0 <= row < h and 0 <= col < w:
                    nei_ids.append(row * w + col)
            tiling.nei_ids = np.array(nei_ids)

            self.tilings_data.append(tiling.center)
            self.tilings_nei_ids[tiling.id].extend(tiling.nei_ids)

            vis_tris = []
            vis_polys = []
            vis_rays = []
            for tri in tiling.vis_tri:
                n_tri = np.array(tri)
                vis_ori = n_tri[0, 0] if n_tri.shape[0] > 0 else None
                vis_poly = []
                for idx in range(n_tri.shape[0]):
                    tri = n_tri[idx]
                    vis_poly.append(tuple(tri[1, :]))
                vis_poly = Polygon(list(OrderedDict.fromkeys(vis_poly)))
                correct_times = 0
                while not vis_poly.is_valid:
                    vis_poly = vis_poly.buffer(0)
                    correct_times += 1
                    if correct_times > 10:
                        raise Exception('vis polygon correction failed!')
                v_ray = []
                # for shapely, the last point of polygon is the start (circle form)
                for v_p in np.array(vis_poly.exterior.coords)[0:-1]:
                    ray = SimpleRay()
                    ray.hit = v_p
                    ray.origin = vis_ori
                    ray.rot_angle = geo.calc_axis_x_angle(ray.hit - vis_ori)
                    v_ray.append(ray)
                # tiling.vis_rays.sort(key=lambda x: x.rot_angle)
                v_ray = tuple(v_ray)

                vis_tris.append(n_tri)
                vis_polys.append(vis_poly)
                vis_rays.append(v_ray)
            tiling.vis_tri = tuple(vis_tris)
            tiling.vis_poly = tuple(vis_polys)
            tiling.vis_rays = tuple(vis_rays)
            if len(tiling.vis_tri) > 1:
                tiling.vis_multi_areas = True
            else:
                tiling.vis_multi_areas = False

            if tiling.type or tiling.cross_bound.shape[0] > 0:
                self.tilings_walkable.append(tiling)
                self.tilings_visitable[tiling.id] = 1
            print('\r{} loading: {:.2f}%'.format(self.name, (count + 1) / len(self.tilings) * 100), end='')
        print()
        self.tilings_data = np.array(self.tilings_data)
        self.enable_visibility()

    def calc_r2mpe_precomputation(self):
        """
        如采用R2mpe，必须在创建物理场景后调用。

        - 如预计算了每个tiling周围360度内离散可见采样点，该方法生成r2mpe需要的可见采样点组织
        - 计算每个采样点静态场景能量

        Returns:

        """
        w, h = self.tilings_shape
        flat_sig = 0.1
        flat_weights = np.zeros((h, w))
        th = (HUMAN_STEP * 3) ** 2

        for idx, tiling in enumerate(self.tilings):
            setattr(tiling, 'flat_weight', 0)  # 本权重用于显示可行区域，所有可行区域阈值相近
            setattr(tiling, 'flat_grad', np.array([0, 0]))
            setattr(tiling, 'vis_grids_ids', [])
            setattr(tiling, 'prob_energy', 0)
            setattr(tiling, 'sur_obs_bound_tiling_attr', [])

            if tiling.type or tiling.cross_bound.shape[0] > 0:
                for v_id, v_grids in enumerate(tiling.vis_grids):
                    v_poly = tiling.vis_poly[v_id]
                    v_ori = tiling.vis_rays[v_id][0].origin
                    v_grids_ids = []
                    for g_ids in v_grids:
                        for gi in g_ids[::2]:  # 全部都检测太多，隔一个点取一个点，降低计算复杂度
                            gc = self.tilings[gi].center
                            if alg.l2_norm_square(gc - v_ori) <= th and v_poly.covers(Point(gc)):
                                v_grids_ids.append(gi)
                    tiling.vis_grids_ids.append(np.array(v_grids_ids))
            r, c = tiling.mat_loc
            if tiling.type:
                dist_sq = alg.l2_norm_square((tiling.nearst_obs_pos - tiling.center) * 0.01)
                tiling.flat_weight = np.exp(-flat_sig / dist_sq)
            else:
                tiling.flat_weight = 0
            flat_weights[r, c] = tiling.flat_weight
            print('\r{} r2mpe precompute: {:.2f}%'.format(self.name, (idx + 1) / len(self.tilings) * 100),
                  end='')
        print()
        flat_weights = img.blur_image(flat_weights, h, w)
        flat_grads = img.calc_img_grad(flat_weights, h, w)
        for i in range(h):
            for j in range(w):
                tiling = self.tilings[i * w + j]
                tiling.flat_weight = flat_weights[i, j]
                tiling.flat_grad = flat_grads[i, j]

        self.tilings_weights = flat_weights
        self.tilings_weights_grad = flat_grads
        print('{} r2mpe precompute: done'.format(self.name))

    def calc_apfs2t_precomputation(self):
        """
        use the way of https://ieeexplore.ieee.org/document/10458316 to calculate the apf score of each sample grid of
        the scene.

         Note: must be used when the scene is initialized, i.e., you must make sure that the attribute 'tilings' contain
         well-processed tilings.

        Returns:

        """
        detect_rays = tuple([geo.rot_vecs((1, 0), ang) for ang in np.linspace(0, np.pi * 2, 30, endpoint=False)])
        # partition_ang = 30
        # detect_rays = tuple(geo.rot_vecs((1, 0), ang * DEG2RAD) for ang in range(0, 360, partition_ang))
        total_count = len(self.tilings)
        values = []
        for tid, tiling in enumerate(self.tilings):
            setattr(tiling, 'apf2st_score', 9999999)
            if tiling.type:
                score = 0
                for ray_dir in detect_rays:
                    inter, inter_dis = geo.calc_ray_poly_intersection(tiling.center, ray_dir, self.poly_contour)
                    score += 100 / inter_dis
                tiling.apf2st_score = score
                values.append(tiling.apf2st_score)
                print('\rprocessing apf-s2t scores: {:.2f}%'.format(tid / total_count * 100), end='')
        print()

    def calc_located_tiling_conv(self, pos) -> Tuple[Tiling, geo.ConvexPoly]:
        w, h = self.tilings_shape
        xc, yc = ((pos - self.tiling_offset) // self.tiling_w).astype(np.int32)
        xc, yc = alg.clamp(xc, 0, w - 1), alg.clamp(yc, 0, h - 1)
        cur_tiling = self.tilings[yc * w + xc]  # 对应虚拟网格的id
        corr_conv_len = len(cur_tiling.corr_conv_ids)
        cur_conv = None
        if corr_conv_len == 0:
            cur_conv = None
        elif corr_conv_len == 1:
            cur_conv = self.conv_polys[cur_tiling.corr_conv_ids[0]]
        else:
            for i in range(corr_conv_len):
                conv = self.conv_polys[cur_tiling.corr_conv_ids[i]]
                if conv.poly_contour.contains(Point(pos)):
                    cur_conv = conv
                    break
        return cur_tiling, cur_conv

    def enable_visibility(self):
        self.vispoly_observer = Observer()
        self.vispoly_observer.scene = self
        self.vispoly_observer.load_walls()

    def update_visibility(self,
                          pos,
                          fwd=None,
                          fov=360,
                          grids_comp=False,
                          realtime_comp=True) -> Tuple[Tuple, List[List[int]]]:
        """

        Args:
            pos:
            fwd:
            fov: 可见区域左右两边视线夹角 in degree
            grids_comp:计算可见区域内离散网格点，当采用离线计算可见区域时，此设置无意义，算法直接返回存储的可见区域离散网格点；反之，根据设置
            实时计算处在可见区域内的所有网格点
            realtime_comp: 采用实时计算可见性，反之使用离线计算的可见区域（基于网格图）

        Returns:
            vis_tris: 可见三角形，每个三角形第一个点一定是可见原点，三角形列表整体按照逆时针排序，按照绕x轴夹角变大方向排列
            vis_grids: 以1度划分360个扇区，记录每个扇区中可见的采样点的id，当grids_comp启用时返回实际值。反之返回空

        """
        pos = np.array(pos)
        w, h = self.tilings_shape
        xc, yc = ((pos - self.tiling_offset) // self.tiling_w).astype(np.int32)
        xc, yc = alg.clamp(xc, 0, w - 1), alg.clamp(yc, 0, h - 1)
        tiling = self.tilings[yc * self.tilings_shape[0] + xc]
        vis_grids = []
        if realtime_comp:
            vis_rays, vis_tris = self.__calc_fov_visibility(pos,
                                                            fwd,
                                                            fov,
                                                            self.vispoly_observer.update_visible_polygon(pos))
            if grids_comp:
                h_fov = fov * DEG2RAD * 0.5
                if fov == 360:  # 如果视野范围是360度，则将朝向标准化为(0,1)，目的是用于预计算可见区域时的方向标准化
                    fwd = (0, 1)
                # 以可见原点提供360个旋转分区，每个分区记录该分区可能跨越的可见三角的下标。每个分区记录1度圆心角的扇形与fwd的夹角，本质是从-180度
                # 开始，到180度结束。0号位置表示-180~-179
                ang_partition = [[] for _ in range(360)]
                vis_grids = [[] for _ in range(360)]
                # 记录当前三角形是否在其射线间边后面有隐藏边，若有，处于射线夹角内的点还需要检测是否处于三角形内，若没有，则必定在三角形内
                tris_hidden_back = []
                npvis_tris = []
                for idx, tri in enumerate(vis_tris):
                    sray, eray = tri
                    np_s, np_e = np.array(sray.hit), np.array(eray.hit)
                    npvis_tris.append(np.array((pos, np_s, np_e)))

                    s_ang = math.floor(geo.calc_angle_bet_vec(np_s - pos, fwd) * RAD2DEG + 180) % 360
                    e_ang = math.ceil(geo.calc_angle_bet_vec(np_e - pos, fwd) * RAD2DEG + 180) % 360

                    if s_ang <= e_ang:
                        ang_idxes = tuple(range(s_ang, e_ang + 1))
                    else:
                        ang_idxes = list(range(s_ang, 360))
                        ang_idxes.extend(range(0, e_ang + 1))
                        ang_idxes = tuple(ang_idxes)

                    for a_idx in ang_idxes:
                        ang_partition[a_idx].append(idx)
                    tris_hidden_back.append(len(sray.pot_coll_walls) > 0 or len(eray.pot_coll_walls) > 0)
                dfs_deque = deque([(tiling, 0)])
                tiling_visitable = self.tilings_visitable.copy()
                while len(dfs_deque) > 0:
                    cur_t, ang_idx = dfs_deque.popleft()
                    vis_grids[ang_idx].append(cur_t.id)
                    for ni in cur_t.nei_ids:
                        if tiling_visitable[ni]:
                            n_tiling = self.tilings[ni]
                            n_dir = n_tiling.center - pos
                            n_ang = geo.calc_angle_bet_vec(n_dir, fwd)
                            possible_in = True if abs(n_ang) <= h_fov else False
                            if possible_in:
                                ang_idx = math.floor(n_ang * RAD2DEG + 180) % 360
                                tri_ids = ang_partition[ang_idx]
                                if len(tri_ids) == 1:
                                    if not tris_hidden_back[tri_ids[0]]:
                                        dfs_deque.append((n_tiling, ang_idx))
                                    elif geo.chk_p_in_conv_simple(n_tiling.center, npvis_tris[tri_ids[0]]):
                                        dfs_deque.append((n_tiling, ang_idx))
                                else:
                                    for tri_id in tri_ids:
                                        if geo.chk_p_in_conv_simple(n_tiling.center, npvis_tris[tri_id]):
                                            dfs_deque.append((n_tiling, ang_idx))
                                            break
                            tiling_visitable[ni] = 0
        else:
            v_id = tiling.obtain_vis_attr_id(pos)
            if v_id is None:
                return (), []
            v_rays = tiling.vis_rays[v_id]
            pos = v_rays[0].origin
            vis_rays, vis_tris = self.__calc_fov_visibility(pos, fwd, fov, v_rays)
            if grids_comp:
                if fov == 360:
                    vis_grids = tiling.vis_grids[v_id]
                else:
                    t_v_grids = tiling.vis_grids[v_id]
                    fwd = (0, 1)
                    vis_grids = [[] for _ in range(360)]
                    s_ray = vis_rays[0]
                    e_ray = vis_rays[-1]
                    s_idx = math.floor(geo.calc_angle_bet_vec(np.array(s_ray.hit) - pos, fwd) * RAD2DEG + 180) % 360
                    e_idx = math.ceil(geo.calc_angle_bet_vec(np.array(e_ray.hit) - pos, fwd) * RAD2DEG + 180) % 360
                    if s_idx <= e_idx:
                        ang_idxes = tuple(range(s_idx, e_idx))
                    else:
                        ang_idxes = list(range(s_idx, 360))
                        ang_idxes.extend(range(0, e_idx))
                        ang_idxes = tuple(ang_idxes)

                    for bound_idx in (0, -1):
                        test_tri = (pos, vis_tris[bound_idx][0].hit, vis_tris[bound_idx][1].hit)
                        a_idx = ang_idxes[bound_idx]
                        for t_id in t_v_grids[a_idx]:
                            if geo.chk_p_in_conv_simple(self.tilings[t_id].center, np.array(test_tri)):
                                vis_grids[a_idx].append(t_id)

                    for a_idx in ang_idxes[1:len(ang_idxes) - 1]:
                        vis_grids[a_idx] = t_v_grids[a_idx]

        vis_tris = tuple((tuple(pos), tri[0].hit, tri[1].hit) for tri in vis_tris)
        return vis_tris, vis_grids

    @staticmethod
    def __calc_fov_visibility(pos, fwd, fov, vis_rays: Sequence[Ray]) -> tuple[tuple[Ray], tuple[tuple[Ray, Ray], ...]]:
        """

        Args:
            pos:
            fwd:
            fov:
            vis_rays:

        Returns:
            vis_vertexes: 可见区域的外轮廓点
            vis_triangles:可见三角形，记录三角形两边（沿可见原点发出），整体按照逆时针排序，三角形按照射线与x轴夹角排序（逆时针）
        """
        vis_tris = []
        if fov < 360 and fwd is not None:
            vis_rays = list(vis_rays)
            h_fov = fov * DEG2RAD * 0.5
            ray_r = geo.rot_vecs(fwd, h_fov)  # 最右侧视线
            ray_l = geo.rot_vecs(fwd, -h_fov)  # 最左侧视线
            r_find, l_find = False, False
            r_pos, l_pos = None, None
            side_rays = []
            for vi in range(-1, len(vis_rays) - 1):
                if not r_find:
                    inter = geo.calc_ray_line_intersection(np.array(vis_rays[vi].hit),
                                                           np.array(vis_rays[vi + 1].hit),
                                                           pos,
                                                           ray_r)
                    if inter is not None:
                        r_pos = inter[-1]
                        r_find = True
                if not l_find:
                    inter = geo.calc_ray_line_intersection(np.array(vis_rays[vi].hit),
                                                           np.array(vis_rays[vi + 1].hit),
                                                           pos,
                                                           ray_l)
                    if inter is not None:
                        l_pos = inter[-1]
                        l_find = True

                if r_find and l_find:
                    break
            for side_pos in (r_pos, l_pos):
                side_ray = Ray()
                side_ray.rot_angle = geo.calc_axis_x_angle(side_pos - pos)
                side_ray.hit = tuple(side_pos)
                vis_rays.append(side_ray)
                side_rays.append(side_ray)
            vis_rays.sort(key=lambda r: r.rot_angle)
            start_idx = vis_rays.index(side_rays[0])
            end_idx = vis_rays.index(side_rays[1])
            if start_idx < end_idx:
                vis_rays = vis_rays[start_idx:end_idx + 1:1]
            else:
                vis_rays = vis_rays[start_idx::] + vis_rays[:end_idx + 1]
            tri_range = range(len(vis_rays) - 1)

        else:
            tri_range = list(range(0, len(vis_rays) - 1))
            tri_range.append(-1)

        for ti in tri_range:
            tri_s = vis_rays[ti]
            tri_e = vis_rays[ti + 1]
            vis_tris.append((tri_s, tri_e))
        return tuple(vis_rays), tuple(vis_tris)  # vis_vertexes逆时针排序，所以vis_tris里三角形顶点排序也是逆时针

    def calc_tiling_diffusion(self, tiling_idx: Tuple, init_fwd, fov=120, depth=6, rot_theta=0) -> Tuple[Tuple, Tuple]:
        """
        计算以像素为单位，fov视角范围内，depth深度内某像素的邻域像素，返回结果以最近向最外，逐层扩散的方式返回邻域像素。例：
            fov = 120
            depth = 6
            rot_theta = 0

            - - - - - p - - - - -
            - - p p p p p p p - -
            - p p p p p p p p p -
            - - p p p p p p p - -
            - - - - p p p - - - -
            - - - - - c - - - - -
            - - - - - - - - - - -
            - - - - - - - - - - -

            其中，p是范围内像素，-是范围外像素

        Args:
            tiling_idx: 中心tiling的行列号（w,h）
            fov:
            init_fwd: 初始正方向
            pixel_width: 像素宽度
            depth: 可观测到的最深像素深度（中心像素depth-1范围）
            rot_theta: 正方向旋转角（角度制），默认以[0,1] 为正方向

        Returns: 左右两个视野半区的tiling编号，每个编号(col,row)

        """

        fov = fov * DEG2RAD
        h_fov = fov * 0.5
        fwd = geo.rot_vecs(init_fwd, rot_theta)
        range_layer1 = [tiling_idx]
        range_layer2 = []
        for i in range(1, depth):
            rans = ((-i, i + 1, 1), (1 - i, i, 1), (i, -i - 1, -1), (i - 1, -i, -1))
            for ri, ran in enumerate(rans):
                for k in range(*ran):
                    if ri == 0:
                        idx = (-i, k)
                    elif ri == 1:
                        idx = (k, i)
                    elif ri == 2:
                        idx = (i, k)
                    else:
                        idx = (k, -i)
                    col = idx[0] + tiling_idx[0]
                    row = idx[1] + tiling_idx[1]
                    if 0 <= col < self.tilings_shape[0] and 0 <= row < self.tilings_shape[1]:
                        tar_vec = np.array(idx)
                        if alg.l2_norm(tar_vec) <= (depth - 1):
                            tar_ang = geo.calc_angle_bet_vec(tar_vec, fwd)
                            if abs(tar_ang) <= h_fov:
                                if tar_ang >= 0:
                                    range_layer1.append((col, row))
                                else:
                                    range_layer2.append((col, row))

        return tuple(range_layer1), tuple(range_layer2)

    def calc_diffusion_dist(self, user_loc, user_fwd, enable_obs_coff=True):
        cur_tiling, _ = self.calc_located_tiling_conv(user_loc)
        sur_tiling_centers = self.tilings_data.copy()
        sur_tiling_probs = np.zeros(len(self.tilings))
        for sur_tiling in self.tilings:
            sur_vec = sur_tiling.center - user_loc
            sur_vel = alg.l2_norm(sur_vec)
            sur_prob = 0
            if sur_tiling.type:
                theta = geo.calc_angle_bet_vec(sur_vec, user_fwd)
                # 将[-pi,pi]作为99%置信区间，防止角度大时能量太小 https://blog.csdn.net/kaede0v0/article/details/113790060
                # rot_prob = np.exp(-0.5 * (theta / np.pi) ** 2 * 4) / ((2 * np.pi) ** 0.5)
                # 将3*human step作为99%置信区间 https://blog.csdn.net/kaede0v0/article/details/113790060
                # mov_prob = np.exp(-0.5 * (sur_vel / self.human_step_single) ** 2 * 9 / 4) / ((2 * np.pi) ** 0.5)
                # sur_prob = rot_prob * mov_prob
                sur_prob = np.exp(
                    -(4.5 * (theta * REV_PI) ** 2 + 0.5 * (sur_vel * REV_HUMAN_STEP) ** 2)) * REV_PI_2
                obs_coff = 1
                if enable_obs_coff and sur_vel > 0 and len(cur_tiling.sur_obs_gids) > 0:
                    norm_vec = sur_vec / sur_vel
                    obs_coff = float('inf')
                    for obs_id in cur_tiling.sur_bound_gids:
                        obs_tiling = self.tilings[obs_id]
                        obs_vec = user_loc - obs_tiling.center
                        obs_d = alg.l2_norm(obs_vec)
                        epsilon = ((obs_d + np.dot(norm_vec, obs_vec)) / obs_d * 0.5) ** (HUMAN_STEP / obs_d)
                        if epsilon < obs_coff:
                            obs_coff = epsilon
                sur_prob *= obs_coff
                sur_prob *= sur_tiling.flat_weight
            sur_tiling_probs[sur_tiling.id] = sur_prob

        return sur_tiling_centers, sur_tiling_probs

    def interpolate_tiling_weight_grad(self, pos, ws=None, gs=None):
        """
        采用双线性插值
        Args:

            pos:
            ws:
            gs:

        Returns:

        """
        xc, yc = ((pos - self.tiling_offset) // self.tiling_w).astype(np.int32)
        cur_tiling = self.tilings[yc * self.tilings_shape[0] + xc]  # 对应虚拟网格的id
        yc, xc = cur_tiling.mat_loc

        x, y = (pos - cur_tiling.center) * self.tiling_w_inv  # col_loc - col_s, row_loc - row_s
        if x >= 0:
            x2 = 1
            c = 1
        else:
            x2 = -1
            c = -1
        if y >= 0:
            y2 = 1
            r = 1
        else:
            y2 = -1
            r = -1
        ye = yc + r if 0 <= yc + r < self.tilings_shape[1] else yc
        xe = xc + c if 0 <= xc + c < self.tilings_shape[0] else xc
        x_cof = np.array([[x2 - x], [x]]) * x2
        y_cof = np.array([[y2 - y, y]]) * y2
        tw = None
        if ws is not None:
            w11 = ws[yc, xc]
            w12 = ws[ye, xc]
            w21 = ws[yc, xe]
            w22 = ws[ye, xe]
            w = np.array([[w11, w21], [w12, w22]])
            tw = (y_cof @ w @ x_cof).squeeze()

        tg = None
        if gs is not None:
            g11x, g11y = gs[yc, xc]
            g12x, g12y = gs[ye, xc]
            g21x, g21y = gs[yc, xe]
            g22x, g22y = gs[ye, xe]
            gx = np.array([[g11x, g21x], [g12x, g22x]])
            gy = np.array([[g11y, g21y], [g12y, g22y]])
            tgx = y_cof @ gx @ x_cof
            tgy = y_cof @ gy @ x_cof
            tg = np.concatenate((tgx, tgy), axis=1).squeeze()

        return tw, tg

    def calc_spatial_complex(self):
        """
        use the metric of arc

        :return:
        """
        total_sc = 0
        total_num = 0
        for tiling in self.tilings:
            if tiling.type:
                nearest_obs = self.tilings[tiling.nearst_obs_gid]
                d = alg.l2_norm(nearest_obs.center - tiling.center)
                total_sc += d
                total_num += 1
        total_sc /= total_num
        return total_sc

    def clear(self):
        super().clear()
        if self.tilings is not None:
            self.tilings = ()
            self.tilings_shape = None
            self.tiling_w = 10
