import time

import numpy as np

from pyrdw.lib.math import *
import pyrdw.lib.math.algebra as alg
from typing import List, Tuple, Set, Sequence, Dict, Optional, Iterator


class Triangle:

    def __init__(self):
        self.vertices = []
        self.poly_contour: Polygon = Polygon()
        self.barycenter = []
        self.in_circle = []
        self.out_edges = []
        self.in_edges = []

    def set_points(self, p1, p2, p3):
        self.vertices = [p1, p2, p3]
        self.barycenter = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3]
        self.in_circle = [0, 0]
        self.out_edges = []
        self.in_edges = []

    def det_common_edge(self, other_t):
        """
        与另一个三角形对比，找共同边

        Args:
            other_t: 另一个三角形

        Returns:

        """
        for i in range(-1, len(self.vertices) - 1):
            e = [self.vertices[i], self.vertices[i + 1]]
            for j in range(-1, len(other_t.vertices) - 1):
                o_e = [other_t.vertices[j], other_t.vertices[j + 1]]
                if chk_edge_same(e, o_e):
                    return e
        return None

    def find_vertex_idx(self, v):
        for i in range(len(self.vertices)):
            if chk_p_same(self.vertices[i], v):
                return i

    def clone(self):
        p1, p2, p3 = self.vertices
        cp1, cp2, cp3 = p1[:], p2[:], p3[:]
        c_tri = Triangle()
        c_tri.set_points(cp1, cp2, cp3)
        c_tri.vertices = pickle.loads(pickle.dumps(self.vertices))
        c_tri.barycenter = pickle.loads(pickle.dumps(self.barycenter))
        c_tri.in_circle = pickle.loads(pickle.dumps(self.in_circle))
        c_tri.out_edges = pickle.loads(pickle.dumps(self.out_edges))
        c_tri.in_edges = pickle.loads(pickle.dumps(self.in_edges))
        return c_tri


class ConvexPoly:

    def __init__(self, vertices=None):
        self.vertices = vertices  # list type
        self.poly_contour: Polygon = Polygon()
        self.center = np.array([0, 0])  # 质心
        self.barycenter = None
        self.cir_circle = []  # 外接圆，形式[x,y],r
        self.in_circle = []  # 内切圆，形式[x,y],r
        self.cir_rect = []  # 最小外界矩形
        self.out_edges = []
        self.in_edges = []
        self.area = 0
        self.out_edges_perimeter = 0
        if self.vertices is not None:
            self.generate_from_poly(self.vertices)

    def generate_from_poly(self, poly_points):
        self.vertices = cmp_convex_vertex_order(poly_points)
        self.poly_contour = Polygon(self.vertices)
        self.center = np.array(self.poly_contour.centroid.coords)[0]
        self.barycenter = calc_poly_barycenter(np.array(poly_points))
        self.cir_circle = calc_poly_min_cir_circle(poly_points)
        self.in_circle = calc_poly_max_in_circle(poly_points)

    def det_common_edge(self, other_p):
        """
        与另一个凸多边形对比，找共同边

        Args:
            other_p: 另一个三角形

        Returns:

        """
        for i in range(-1, len(self.vertices) - 1):
            e = [self.vertices[i], self.vertices[i + 1]]
            for j in range(-1, len(other_p.vertices) - 1):
                o_e = [other_p.vertices[j], other_p.vertices[j + 1]]
                if chk_edge_same(e, o_e):
                    return e
        return None

    def find_vertex_idx(self, v):
        for i in range(len(self.vertices)):
            if chk_p_same(self.vertices[i], v):
                return i

    def clone(self):
        c_poly = ConvexPoly()
        c_poly.vertices = pickle.loads(pickle.dumps(self.vertices))
        c_poly.in_circle = pickle.loads(pickle.dumps(self.in_circle))
        c_poly.center = self.center.copy()
        c_poly.barycenter = pickle.loads(pickle.dumps(self.barycenter))
        c_poly.out_edges = pickle.loads(pickle.dumps(self.out_edges))
        c_poly.in_edges = pickle.loads(pickle.dumps(self.in_edges))
        return c_poly


def rot_vecs(v, ang) -> np.ndarray:
    """
    绕(0,0)点旋转指定向量

    Args:
        v: 指定向量
        ang: 旋转角(弧度制)，+顺时针，-逆时针

    Returns: 旋转后向量

    """

    sin_t, cos_t = np.sin(ang), np.cos(ang)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return np.dot(v, rot_mat)


def norm_vec(v):
    """
    归一化向量
    Args:
        v: 向量

    Returns:
        归一化向量
    """
    v_o = np.array(v)
    v_l = (v_o[0] ** 2 + v_o[1] ** 2) ** 0.5
    if v_l == 0:
        return np.array([0, 0])
    return np.divide(v_o, v_l)


def calc_axis_x_angle(p):
    """
    计算目标点的x轴转角，逆时针为正，（1,0）为0度

    Args:
        p:

    Returns:
        in degree


    """
    return math.atan2(p[1], p[0]) * RAD2DEG % 360


def calc_poly_area(poly: List | Tuple | np.ndarray | Polygon):
    """
    计算多边形面积(凹凸均可)

    Args:
        poly:

    Returns:

    """
    return Polygon(poly).area


def calc_poly_barycenter(poly):
    """
    计算多边形重心

    Args:
        poly: 多边形顶点列表,例如：[[1,2],
                                [3,4],
                                [2,3]]

    Returns:
        重心，以list表示
    """
    row, col = poly.shape
    index = range(row - 2)
    area_sum = 0
    center = np.array([0, 0])
    for i in index:
        tri_p = poly[[0, i + 1, i + 2]]
        tri_a = calc_poly_area(tri_p)
        area_sum += tri_a
        tri_c = np.sum(tri_p, axis=0)
        center = center + tri_c * tri_a

    return np.divide(center, 3 * area_sum)


def calc_poly_centroid(poly):
    """
    求解凸多边形质心

    Args:
        poly:

    Returns:

    """
    row, col = poly.shape
    if row < 2:
        return None
    else:
        return np.divide(np.sum(poly, axis=0), row)


def calc_angle_bet_vec(v_base, v_target):
    """
    返回两不为0向量间夹角，从v_base转过所要计算的角度到v_target，以弧度制

    Args:
        v_base: 旋转向量
        v_target: 目标向量

    Returns:
        旋转角 0 - PI 顺时针转角 -PI - 0 逆时针转角
    """
    if (v_base[0] == 0 and v_base[1] == 0) or (v_target[0] == 0 and v_target[1] == 0):
        return 0
    vbtan = np.arctan2(v_base[1], v_base[0])
    vttan = np.arctan2(v_target[1], v_target[0])
    ang_base = vbtan if vbtan > 0 else vbtan + PI_2
    ang_tar = vttan if vttan > 0 else vttan + PI_2
    turn_ang = ang_base - ang_tar
    if turn_ang > PI:
        turn_ang -= PI_2
    elif turn_ang < -PI:
        turn_ang += PI_2
    return turn_ang


def calc_cir_rect(poly):
    """
    计算给定标准旋转状态下多边形最小外接矩形

    Args:
        poly: 多边形顶点列表，ndarray

    Returns:
        最小外接矩形，宽，高
    """
    [x_min, y_min] = np.min(poly, axis=0)
    [x_max, y_max] = np.max(poly, axis=0)
    min_rect = np.array([[x_min, y_min],
                         [x_max, y_min],
                         [x_max, y_max],
                         [x_min, y_max]])
    return min_rect, x_max - x_min, y_max - y_min


def calc_poly_min_cir_rect(poly):
    """
    计算给定多边形的最小外接矩形

    Args:
        poly: 多边形定点序列，二维数组形式

    Returns:
        最小外接矩形，面积，多边形对应旋转角度
    """

    N, d = poly.shape
    if N < 3 or d != 2:
        raise ValueError
    rect_min, w_min, h_min = calc_cir_rect(poly)
    rad_min = 0.
    area_min = w_min * h_min
    rad = []
    for i in range(N):
        vector = poly[i - 1] - poly[i]
        rad.append(np.arctan(vector[1] / (vector[0] + EPS)))
    for r in rad:
        new_poly = rot_vecs(poly, r)
        rect, w, h = calc_cir_rect(new_poly)
        area = w * h
        if area < area_min:
            rect_min, area_min, w_min, h_min, rad_min = rect, area, w, h, -r
    min_rect_r = rot_vecs(rect_min, rad_min)
    return min_rect_r, w_min, h_min, rad_min


def calc_poly_min_cir_circle(poly: list):
    """
    计算多边形的最小外接圆

    Args:
        poly:

    Returns:
        [[xc,yc],r]
    """
    tar_p = pickle.loads(pickle.dumps(poly))
    cur_p_set = []
    p1, p2 = tar_p.pop(), tar_p.pop()
    cur_p_set.append(p1)
    cur_p_set.append(p2)
    cur_circle = [[(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5], alg.l2_norm(np.array(p1) - np.array(p2)) * 0.5]
    while len(tar_p) > 0:
        pt = tar_p.pop()
        if not chk_p_in_circle(pt, cur_circle[0], cur_circle[1]):
            p_num = len(cur_p_set)
            min_r = float('inf')
            min_cir = []
            for i in range(p_num - 1):
                p1 = cur_p_set[i]
                for j in range(i + 1, p_num):
                    p2 = cur_p_set[j]
                    temp_circle = calc_tri_min_cir_circle(np.array(p1), np.array(p2), np.array(pt))
                    in_temp = True
                    for tp in cur_p_set:
                        if not chk_p_in_circle(tp, temp_circle[0], temp_circle[1]):
                            in_temp = False
                            break
                    if in_temp:
                        if temp_circle[1] <= min_r:
                            min_cir = temp_circle
                            min_r = temp_circle[1]
            cur_circle = min_cir
        cur_p_set.append(pt)
    return cur_circle


def calc_tri_min_cir_circle(p1, p2, p3):
    """
    求三角形最小外接圆

    Args:
        p1:
        p2:
        p3:

    Returns:
        [[xc,yc],r]
    """
    d1 = alg.l2_norm(p1 - p2)
    d2 = alg.l2_norm(p1 - p3)
    d3 = alg.l2_norm(p2 - p3)
    d_max = max(d1, d2, d3)
    if d1 == d_max:
        xt, yt, rt = (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, d1 * 0.5
        other_p = p3
    elif d2 == d_max:
        xt, yt, rt = (p1[0] + p3[0]) * 0.5, (p1[1] + p3[1]) * 0.5, d2 * 0.5
        other_p = p2
    else:
        xt, yt, rt = (p2[0] + p3[0]) * 0.5, (p2[1] + p3[1]) * 0.5, d3 * 0.5
        other_p = p1
    d2other = alg.l2_norm(np.array([xt, yt]) - other_p)
    if d2other <= rt:
        x0, y0, r = xt, yt, rt
    else:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x0 = (y3 - y2) * (x2 ** 2 + y2 ** 2 - x1 ** 2 - y1 ** 2) - (y2 - y1) * (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2)
        x0 = x0 / ((y3 - y2) * (x2 - x1) - (y2 - y1) * (x3 - x2)) * 0.5
        y0 = (x2 - x1) * (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2) - (x3 - x2) * (x2 ** 2 + y2 ** 2 - x1 ** 2 - y1 ** 2)
        y0 = y0 / ((y3 - y2) * (x2 - x1) - (y2 - y1) * (x3 - x2)) * 0.5
        r = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return [[x0, y0], r]


def calc_poly_max_in_circle(poly: list):
    tar_poly = pickle.loads(pickle.dumps(poly))
    x_sub, y_sub = 20, 20
    stop_thre = 0.1
    x_min, x_max = float('inf'), 0
    y_min, y_max = float('inf'), 0
    for p in tar_poly:
        x, y = p
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    bound = [x_min, x_max, y_min, y_max]
    interval = (2 ** 0.5) * 2

    while True:
        c_center, radius = calc_poly_in_circle(tar_poly, bound, x_sub, y_sub)
        cx, cy = c_center
        fit_tmp = (bound[1] - bound[0]) / interval
        bound[1] = cx + fit_tmp
        bound[0] = cx - fit_tmp
        fit_tmp = (bound[3] - bound[2]) / interval
        bound[3] = cy + fit_tmp
        bound[2] = cy - fit_tmp
        if (bound[1] - bound[0]) < stop_thre or (bound[3] - bound[2]) < stop_thre:
            break

    return c_center, radius


def calc_poly_in_circle(poly: list, out_bound, x_sub, y_sub):
    poly = np.array(poly)
    c_center = [0, 0]
    x_inc = (out_bound[1] - out_bound[0]) / x_sub
    y_inc = (out_bound[3] - out_bound[2]) / y_sub
    max_dis = 0
    for i in range(x_sub):
        x_temp = out_bound[0] + i * x_inc
        for j in range(y_sub):
            y_temp = out_bound[2] + j * y_inc
            if chk_p_in_conv_simple([x_temp, y_temp], poly):
                dis_temp, _ = calc_point_mindis2poly([x_temp, y_temp], poly)
                if dis_temp > max_dis:
                    max_dis = dis_temp
                    c_center = [x_temp, y_temp]
    return c_center, max_dis


def calc_point_mindis2bound(pos, bounds):
    min_dis = float('inf')
    in_bound = True
    for bound in bounds:
        in_cur_bound = False
        b_ps = bound.points
        b_ps_num = len(b_ps)
        intersect_num = 0
        for i in range(-1, b_ps_num - 1):
            b_s, b_e = b_ps[i], b_ps[i + 1]
            if chk_right_ray_line_cross(pos, b_s, b_e):
                intersect_num = intersect_num + 1
            p2curb = calc_point_mindis2line(pos, b_s, b_e)
            # 如果距离当前边界过进，则认为发生边界碰撞，直接触发出界
            if p2curb <= min_dis:
                min_dis = p2curb
        if intersect_num % 2 != 0:
            in_cur_bound = True
        if in_cur_bound and not bound.is_out_bound:
            in_bound = False
            break
        elif not in_cur_bound and bound.is_out_bound:
            in_bound = False
            break
    if in_bound:
        return min_dis
    else:
        return None


def calc_point_mindis2poly(pos, poly: np.ndarray):
    row, col = poly.shape
    min_dis = float('inf')
    mp1, mp2 = None, None
    for i in range(-1, row - 1):
        p1, p2 = poly[i], poly[i + 1]
        dis, r, t = calc_point_pro2line(pos, p1, p2)
        if dis < min_dis:
            min_dis = dis
            mp1, mp2 = p1, p2
    return min_dis, [mp1, mp2]


def calc_point_pro2line(p, l_s, l_e):
    """
    计算点在指定线段上的投影

    Args:
        p: 点
        l_s: 线段起点
        l_e: 线段终点

    Returns:
        投影距离，投影点，投影点在线段上的t值
    """

    point_loc = np.array(p)
    line_s_loc = np.array(l_s)
    line_e_loc = np.array(l_e)
    line_vec = line_e_loc - line_s_loc
    r_dir_normal = norm_vec(line_vec)
    t = np.dot(point_loc - line_s_loc, r_dir_normal)
    result = np.multiply(t, r_dir_normal) + l_s
    v = result - point_loc
    distance = (v[0] ** 2 + v[1] ** 2) ** 0.5
    if line_vec[0] != 0:
        t_t = t * r_dir_normal[0] / line_vec[0]
    elif line_vec[1] != 0:
        t_t = t * r_dir_normal[1] / line_vec[1]
    else:
        t_t = None
        distance = None
        result = None
    return distance, result, t_t


def calc_p_project_on_line(point: np.ndarray,
                           line: np.ndarray):
    """
    计算点在线段上的投影，若投影点不在线段中，则返回距离point最近的线段端点。

    Args:
        point: 要投影的点
        line: 线段

    Returns:投影点, t值（若长度为0，则为None）

    """

    v = line[1] - line[0]  # 计算线段的向量
    w = point - line[0]  # 计算点到线段起点的向量
    c1, c2 = w @ v, v @ v  # 计算投影参数
    if c2 == 0:  # 如果线段长度为零（起点和终点相同），则返回起点
        return line[0], None  # 计算投影点的参数t
    t = c1 / c2
    # 确保投影点在线段上
    if t < 0:
        return line[0], t
    elif t > 1:
        return line[1], t
    else:  # 计算投影点的坐标
        return line[0] + t * v, t


def calc_point_mindis2line(p, l_s, l_e):
    p_loc = np.array(p)
    s_loc = np.array(l_s)
    e_loc = np.array(l_e)
    cos1 = np.dot(p_loc - s_loc, e_loc - s_loc)
    cos2 = np.dot(p_loc - e_loc, s_loc - e_loc)
    if cos1 * cos2 >= 0:  # p在直线的投影点在线段起点和终点之间（考虑了与起始点重合的情况）
        # 采用直线一般式Ax+By+C=0表示线段所在直线，继而求出点到直线投影距离
        a = e_loc[1] - s_loc[1]
        b = s_loc[0] - e_loc[0]
        c = e_loc[0] * s_loc[1] - s_loc[0] * e_loc[1]
        d = (a ** 2 + b ** 2) ** 0.5
        return np.fabs(a * p_loc[0] + b * p_loc[1] + c) / d
    else:  # 点的投影在线段两端之外，此时直接求点到两端的最小值即可
        ps = p_loc - s_loc
        d1 = (ps[0] ** 2 + ps[1] ** 2) ** 0.5
        es = p_loc - e_loc
        d2 = (es[0] ** 2 + es[1] ** 2) ** 0.5
        if d1 < d2:
            return d1
        else:
            return d2


def calc_lines_intersect(l1_s, l1_e, l2_s, l2_e):
    """
    线段求交点

    Args:
        l1_s: 线段1起点
        l1_e: 线段1终点
        l2_s: 线段2起点
        l2_e: 线段2终点

    Returns:
        交点
    """
    l1_s = np.array(l1_s)
    l1_e = np.array(l1_e)
    l2_s = np.array(l2_s)
    l2_e = np.array(l2_e)
    l1_v = l1_e - l1_s
    l2_v = l2_e - l2_s
    i_type = -1  # -2 参数错误,  -1 - 无交点， 0 - 因共线有交集， 1 - 有一个交点
    intersect = []
    extend_info = []
    if (l1_v[0] == 0 and l1_v[1] == 0) or (l2_v[0] == 0 and l2_v[1] == 0):  # 某线段是一个点，不能进行判断
        i_type = -2
        intersect = None
        extend_info = None
    elif alg.cross(l1_v, l2_v) != 0:  # 两线段不共线
        a, b = l1_v[0], l1_v[1]
        c, d = l2_v[0], l2_v[1]
        e, f = l2_s[0] - l1_s[0], l2_s[1] - l1_s[1]
        s = (e * d - c * f) / (a * d - c * b)
        t = (e * b - a * f) / (a * d - c * b)
        if 0 <= s <= 1 and 0 <= t <= 1:  # 有交点
            i_type = 1
            intersect = np.multiply(l1_v, s) + l1_s
            extend_info = (s, t)
        else:  # 无交点
            i_type = -1
            intersect = None
            extend_info = None
    else:  # 共线
        l3_v = l2_e - l1_s
        if alg.cross(l1_v, l3_v) != 0:  # 两线段平行，不可能有交点
            i_type = -1
            intersect = None
            extend_info = None
        else:  # 两线段在同一直线上
            pass  # ----------------------------------------------------------------------------------------------------

    return i_type, intersect, extend_info


def calc_nonparallel_lines_intersect(l1_s, l1_e, l2_s, l2_e):
    """
    非共线线段求交点，若共线则无交点,该方法用于计算两凸多边形交集

    Args:
        l1_s: 线段1起点
        l1_e: 线段1终点
        l2_s: 线段2起点
        l2_e: 线段2终点

    Returns:
        t (# 1 - 有一个交点, 0- 无交点), p (交点), info
    """
    l1_s = np.array(l1_s)
    l1_e = np.array(l1_e)
    l2_s = np.array(l2_s)
    l2_e = np.array(l2_e)
    l1_v = l1_e - l1_s
    l2_v = l2_e - l2_s

    if alg.cross(l1_v, l2_v) != 0:  # 两线段不共线
        a, b = l1_v
        c, d = l2_v
        e, f = l2_s - l1_s
        m = a * d - c * b
        s = (e * d - c * f) / m
        t = (e * b - a * f) / m
        if 0 <= s <= 1 and 0 <= t <= 1:  # 有交点
            return 1, (np.multiply(l1_v, s) + l1_s).tolist(), (s, t)
    return 0, None, None


def calc_ray_line_intersection(l_s: np.ndarray,
                               l_e: np.ndarray,
                               r_s: np.ndarray,
                               r_dir: np.ndarray) -> None | np.ndarray:
    """
    检测射线与指定线段是否相交

    Args:
        l_s:
        l_e:
        r_s:
        r_dir:射线方向

    Returns:
        相交类型，交点
        - None 无交点
        - np.ndarry [num * 2]
    """

    line = LineString([l_s, l_e])
    ray = LineString([r_s, r_s + norm_vec(r_dir) * 1e7])
    inter = line.intersection(ray)

    if inter.is_empty:
        return None
    else:
        points = np.array(inter.coords)
        if points.shape[0] > 1:
            p1 = points[0]
            p2 = points[1]
            if alg.l2_norm(p1 - r_s) > alg.l2_norm(p2 - r_s):  # p1 closer
                points = np.array([p2, p1])
        return points
    # l_left = l_s - r_s
    # l_right = l_e - r_s
    # if alg.cross(l_left, l_right) == 0:  # 射线与线段共线
    #     if np.dot(l_left, l_right) <= 0:  # 射线起点在线段中
    #         if np.dot(r_dir, l_right) >= 0:  # 如果线段终点方向与射线方向相同，相交区域为射线起点到线段终点
    #             return np.array((r_s, l_e))
    #         else:
    #             return np.array((r_s, l_s))
    #     elif np.dot(r_dir, l_left) >= 0:  # 射线起点在线段外，且线段在射线方向，相交区域为线段
    #         return np.array((l_s, l_e))
    #     else:  # 射线起点在线段外，且线段反方向，没有交点
    #         return None
    # else:
    #     r_p = r_s + r_dir
    #     center = (l_s + l_e) * 0.5
    #     v1, v2 = r_p - r_s, center - r_s
    #     rela1 = alg.cross(v1, r_p - l_s) * alg.cross(v2, center - l_s)
    #     rela2 = alg.cross(v1, r_p - l_e) * alg.cross(v2, center - l_e)
    #
    #     if rela1 >= 0 and rela2 >= 0:
    #         xa, ya = l_s
    #         xb, yb = l_e
    #         xe, ye = r_p
    #         xd, yd = r_s
    #         t = (xa * yb - xb * ya + (ya - yb) * xd + (xb - xa) * yd) / ((xe - xd) * (yb - ya) + (ye - yd) * (xa - xb))
    #         return np.array([t * v1 + r_s])
    #     else:
    #         return None


def calc_ray_poly_intersection_bound(pos: np.ndarray,
                                     fwd: np.ndarray,
                                     bounds) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    """
    检测射线与场景边界最近交点

    Args:
        pos:
        fwd:
        bounds:

    Returns:
        交点，最近距离，相交边界
    """

    min_dis = float('inf')
    min_inter = None
    min_bound = []
    for b in bounds:
        for i in range(b.points_num):
            contour_s = np.array(b.points[i - 1])
            contour_e = np.array(b.points[i])
            inter = calc_ray_line_intersection(contour_s, contour_e, pos, fwd)
            if inter is not None:
                if inter.shape[0] == 1:
                    d = alg.l2_norm(inter[0] - pos)
                    if d < min_dis:
                        min_dis = d
                        min_inter = inter[0]
                        min_bound = (contour_s.copy(), contour_e.copy())
                else:
                    inter1, inter2 = inter
                    d1 = alg.l2_norm(inter1 - pos)
                    d2 = alg.l2_norm(inter2 - pos)
                    d, inter = (d1, inter1) if d1 < d2 else (d2, inter2)
                    if d < min_dis:
                        min_dis = d
                        min_inter = inter
                        min_bound = (contour_s.copy(), contour_e.copy())

    return min_inter, min_dis, min_bound


def calc_ray_poly_intersection(pos: np.ndarray,
                               fwd: np.ndarray,
                               poly: Polygon) -> Tuple[np.ndarray, float]:
    """
    Calculate the intersection between a ray and a polygon. Note that the ray origin is excluded from results.

    Args:
        pos:
        fwd:
        poly:

    Returns:
        - intersection point, distance
    """
    ray = LineString([pos, pos + norm_vec(fwd) * 1e7])  # use a long line to represent ray
    inter = ray.intersection(poly)
    if inter.is_empty:
        return None, 0
    else:
        op = Point(pos)
        if isinstance(inter, Point) or isinstance(inter, LineString):
            pot_points = np.array(inter.coords)
        else:
            pot_points = np.concatenate([np.array(g.coords) for g in inter.geoms], axis=0)
        inter_p = None
        min_dis = float('inf')
        for s_p in pot_points:
            p = Point(s_p)
            if p.equals(op):
                continue
            else:
                dis = alg.l2_norm(s_p - pos)
                if dis < min_dis:
                    min_dis = dis
                    inter_p = s_p
        return inter_p, min_dis


def calc_square_bound_cross(center, width, boundaries):
    """
    获取指定正方形区域与区域边界相交的轮廓线

    Args:
        center: 正方形中心
        width: 正方形宽度
        boundaries: 指定区域

    Returns:
        返回与正方形相交的边界轮廓线ndarray = N * 2 * 2，N为轮廓线个数，2、2分别对应某条轮廓线起始点和点的xy坐标
    """
    c_x, c_y = center
    h_w = 0.5 * width
    s_points = [[c_x - h_w, c_y - h_w],
                [c_x + h_w, c_y - h_w],
                [c_x + h_w, c_y + h_w],
                [c_x - h_w, c_y + h_w]]
    cross = []
    for boundary in boundaries:
        walls = boundary.points
        walls_num = boundary.points_num
        for i in range(-1, walls_num - 1):
            sx, sy = walls[i]
            ex, ey = walls[i + 1]
            line = [sx, sy, ex, ey]
            if chk_line_rect_cross(line, s_points):
                cross.append(np.array([walls[i], walls[i + 1]]))
    return np.array(cross)


def chk_ps_on_line_side(p1, p2, l_s, l_e):
    """
    检测两点是否位于线段同侧

    Args:
        p1: 点1
        p2: 点2
        l_s: 线段起点
        l_e: 线段终点

    Returns:
        同侧-1 不同侧-0
    """
    return alg.cross(p1 - l_s, p1 - l_e) * alg.cross(p2 - l_s, p2 - l_e) > 0


def chk_lines_cross(l1_s, l1_e, l2_s, l2_e) -> bool:
    """
    线段求交

    Args:
        l1_s: 线段1起点
        l1_e: 线段1终点
        l2_s: 线段2起点
        l2_e: 线段2终点

    Returns:
        是否相交
    """
    # AABB包围盒快速筛选
    if max(l1_s[0], l1_e[0]) >= min(l2_s[0], l2_e[0]) and max(l2_s[0], l2_e[0]) >= min(l1_s[0], l1_e[0]) and \
            max(l1_s[1], l1_e[1]) >= min(l2_s[1], l2_e[1]) and max(l2_s[1], l2_e[1]) >= min(l1_s[1], l1_e[1]):
        nl1_s = np.array(l1_s)
        nl1_e = np.array(l1_e)
        nl2_s = np.array(l2_s)
        nl2_e = np.array(l2_e)
        return np.cross(nl2_s - nl1_s, nl2_e - nl1_s) * np.cross(nl2_s - nl1_e, nl2_e - nl1_e) <= 0 and \
            np.cross(nl1_s - nl2_s, nl1_e - nl2_s) * np.cross(nl1_s - nl2_e, nl1_e - nl2_e) <= 0
    else:
        return False


def chk_line_bound_cross(l1_s, l1_e, bounds):
    """
    检验线段与指定区域边界是否相交

    Args:
        l1_s: 线段起点
        l1_e: 线段终点
        bounds: 边界

    Returns:
        是否相交
    """
    for bound in bounds:
        b_ps = bound.points
        b_ps_num = len(b_ps)
        for i in range(-1, b_ps_num - 1):
            wall_s = b_ps[i]
            wall_e = b_ps[i + 1]
            intersects = chk_lines_cross(l1_s, l1_e, wall_s, wall_e)
            if intersects:
                return True
    return False


def chk_line_rect_cross(line, rect):
    """
    检查线段与矩形是否相交,包括包含、相交关系

    Args:
        line: 线段 [lsx,lsy,lex,ley]
        rect: 矩形 [[left-bottom],
                   [right-bottom],
                   [right-top],
                   [left-top]]

    Returns:

    """
    l_s = np.array(line[:2])
    l_e = np.array(line[2:4])
    left = rect[0][0]
    right = rect[2][0]
    bottom = rect[0][1]
    top = rect[2][1]
    if (left <= l_s[0] <= right and bottom <= l_s[1] <= top) or (left <= l_e[0] <= right and bottom <= l_e[1] <= top):
        return 1
    else:
        if chk_lines_cross(l_s, l_e, rect[0], rect[2]) or chk_lines_cross(l_s, l_e, rect[1], rect[3]):
            return 1
        else:
            return 0


def chk_right_ray_line_cross(r_s, l_s, l_e):
    """
    以给定点为起点，向右发射的射线是否与给定线段有交点，交点与给定点间距小于指定阈值也记为未相交。

    Args:
        r_s: 射线指定点
        l_s: 线段起点
        l_e: 线段终点

    Returns:
        是否相交
    """

    if l_s[1] == l_e[1]:  # 线段与x轴平行
        return False
    if l_s[1] > r_s[1] and l_e[1] > r_s[1]:  # 线段在射线上方
        return False
    if l_s[1] < r_s[1] and l_e[1] < r_s[1]:  # 线段在射线下方
        return False
    if l_s[1] == r_s[1] and l_e[1] > r_s[1]:
        return False
    if l_e[1] == r_s[1] and l_s[1] > r_s[1]:
        return False
    if l_s[0] < r_s[0] and l_e[0] < r_s[0]:
        return False
    x_seg = l_e[0] - (l_e[0] - l_s[0]) * (l_e[1] - r_s[1]) / (l_e[1] - l_s[1])  # 求交
    if x_seg < r_s[0]:
        return False
    else:
        return True


def chk_p_in_poly(pos: List | Tuple | np.ndarray | Point,
                  poly: List | Tuple | np.ndarray | Polygon):
    if not isinstance(pos, Point):
        pos = Point(pos)
    if not isinstance(poly, Polygon):
        poly = Polygon(poly)
    return poly.covers(pos)


def chk_p_in_bound(pos, bounds, danger_dis=0):
    """
    检测指定点是否在边界内

    Args:
        pos: 点位置
        bounds: 区域边界
        danger_dis: : 危险距离，若距离某条边小于该阈限，直接触发边界碰撞，导致出界

    Returns:
        若在区域内，则True
    """
    in_bound = True
    for bound in bounds:
        in_cur_bound = False
        b_ps = bound.points
        b_ps_num = len(b_ps)
        intersect_num = 0
        for i in range(-1, b_ps_num - 1):
            b_s, b_e = b_ps[i], b_ps[i + 1]
            if chk_right_ray_line_cross(pos, b_s, b_e):
                intersect_num = intersect_num + 1
            # 如果距离当前边界过近，则认为发生边界碰撞，直接触发出界
            if calc_point_mindis2line(pos, b_s, b_e) <= danger_dis:
                return False
        if intersect_num % 2 != 0:
            in_cur_bound = True
        if in_cur_bound and not bound.is_out_bound:
            in_bound = False
            break
        elif not in_cur_bound and bound.is_out_bound:
            in_bound = False
            break

    return in_bound


def chk_p_in_conv(pos, poly: list, poly_in_circle):
    """
    计算给定点是否在指定的凸多边形内，在边界上也算在多边形内，需要多边形必须是逆时针排序

    Args:

        pos:
        poly:
        poly_in_circle:

    Returns:
    """

    c, r = poly_in_circle
    c = np.array(c)
    v = pos - c
    if v[0] ** 2 + v[1] ** 2 <= r ** 2:
        return True
    else:
        n_poly = np.array(poly)
        n_poly = n_poly - pos
        poly_num = len(poly)
        for i in range(-1, poly_num - 1):
            pv1, pv2 = n_poly[i], n_poly[i + 1]
            cross = alg.cross(pv1, pv2)
            if cross < 0:  # 点在多边形边右侧，代表在多边形之外
                return False
            elif cross == 0 and np.dot(pv1, pv2) <= 0:  # 点p在直线v1v2上，并且在线段v1v2之间，则直接判定在多边形内
                return True
        return True


def chk_p_in_conv_simple(pos: np.ndarray, poly: np.ndarray):
    """
    计算给定点是否在指定的凸多边形内，在边界上也算在多边形内，需要多边形必须是逆时针排序

    Args:
        pos: (x,y)
        poly:((x1,y1),(x2,y2),.....(xn,yn))

    Returns:

    """

    row, col = poly.shape
    p_poly = poly - pos
    for i in range(-1, row - 1):
        pv1, pv2 = p_poly[i], p_poly[i + 1]
        cross = alg.cross(pv1, pv2)
        if cross < 0:  # 点在多边形边右侧，代表在多边形之外
            return False
        elif cross == 0 and np.dot(pv1, pv2) <= 0:  # 点p在直线v1v2上，并且在线段v1v2之间，则直接判定在多边形内
            return True
    return True


def chk_p_in_tiling_simple(pos, tiling):
    return tiling.rect[0][0] <= pos[0] <= tiling.rect[1][0] and tiling.rect[0][1] <= pos[1] <= tiling.rect[2][1]


def chk_p_in_circle(pos, c_center, c_r):
    vec = np.array(pos) - np.array(c_center)
    return (alg.l2_norm(vec) - c_r) <= EPS


def chk_square_bound_cross(center, width, boundaries):
    """
    检测指定正方形边界是否与区域边界相交

    Args:
        center: 正方形中心
        width: 正方形宽度
        boundaries: 指定区域

    Returns:
        若相交，则True
    """
    intersect_boundary = False
    c_x = center[0]
    c_y = center[1]
    h_w = 0.5 * width
    s_points = [[c_x - h_w, c_y - h_w], [c_x + h_w, c_y - h_w],
                [c_x + h_w, c_y - h_w], [c_x + h_w, c_y + h_w],
                [c_x + h_w, c_y + h_w], [c_x - h_w, c_y + h_w],
                [c_x - h_w, c_y + h_w], [c_x - h_w, c_y - h_w]]
    for boundary in boundaries:
        walls = boundary.points
        walls_num = boundary.points_num
        for i in range(-1, walls_num - 1):
            wall_s = walls[i]
            wall_e = walls[i + 1]
            if chk_lines_cross(s_points[0], s_points[1], wall_s, wall_e):
                intersect_boundary = True
                break
            elif chk_lines_cross(s_points[2], s_points[3], wall_s, wall_e):
                intersect_boundary = True
                break
            elif chk_lines_cross(s_points[4], s_points[5], wall_s, wall_e):
                intersect_boundary = True
                break
            elif chk_lines_cross(s_points[6], s_points[7], wall_s, wall_e):
                intersect_boundary = True
                break
        if intersect_boundary:
            break
    return intersect_boundary


def chk_poly_concavity(poly: list):
    """
    判断poly是否是凸多边形，需要输入多边形顶点逆时针序列

    Args:
        poly:

    Returns:

    """

    poly_num = len(poly)
    for i in range(0, poly_num):
        v_f = poly[i - 1]
        v = poly[i]
        v_n = poly[(i + 1) % poly_num]
        vec1 = np.array([v[0] - v_f[0], v[1] - v_f[1]])
        vec2 = np.array([v_n[0] - v[0], v_n[1] - v[1]])
        if calc_angle_bet_vec(vec2, vec1) <= 0:  # 当前拐角不是凸的
            return False
    return True


def chk_convs_intersect(poly1: list, poly1_in_cir, poly2: list, poly2_in_cir):
    """
    计算两个凸多边是否交集
    Args:
        poly1:
        poly1_in_cir:
        poly2:
        poly2_in_cir:

    Returns:
    """
    if len(poly1) < 3 or len(poly2) < 3:
        return False
    else:
        poly1_num, poly2_num = len(poly1), len(poly2)
        total_set = []
        for p in poly2:
            if chk_p_in_poly(p, poly1):
                total_set.append(p)
        if len(total_set) >= 3:
            return True
        for i in range(-1, poly1_num - 1):  # 先检测poly1的所有顶点在不在poly2中,同时把poly1与poly2边的交点得到
            po1v1, po1v2 = poly1[i], poly1[i + 1]
            if chk_p_in_poly(po1v2, poly2):
                can_add = True
                for ep in total_set:
                    if abs(ep[0] - po1v2[0]) < EPS and abs(ep[1] - po1v2[1]) < EPS:  # 判断两点是否相同，去重复
                        can_add = False
                        break
                if can_add:
                    total_set.append(po1v2)
                    if len(total_set) >= 3:
                        return True
            for j in range(-1, poly2_num - 1):
                po2v1, po2v2 = poly2[j], poly2[j + 1]
                # 这里特别注意，由于交点坐标精度问题，可能出现极小的误差，这里需要进一步处理一下，忽略误差
                i_t, i_p, _ = calc_nonparallel_lines_intersect(po1v1, po1v2, po2v1, po2v2)
                # 仅考察poly1和poly2非共线边的交叉点,原因是若两边共线，无非就是重合或完全不相交
                # 若重合，则重合点在之前判断多边形顶点是否在另一多边形内时就会找到
                if i_t > 0:
                    can_add = True
                    for ep in total_set:
                        if abs(ep[0] - i_p[0]) < EPS and abs(ep[1] - i_p[1]) < EPS:  # 判断两点是否相同，去重复
                            can_add = False
                            break
                    if can_add:
                        total_set.append(i_p)
                        if len(total_set) >= 3:
                            return True
        return False


def chk_points_clockwise(points) -> int:
    """
            Check if sequence of points is oriented in clockwise order.

            Args:
                points:

            Returns:
                1 - clockwise
                0 - collinear
                -1 - counterclockwise
    """
    p1, p2, p3 = points[0], points[1], points[2]
    clockwise = ((p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1]))
    if clockwise > 0:
        return 1
    elif clockwise == 0:
        return 0
    else:
        return -1


def chk_ray_rect_AABB(r_s: Tuple | np.ndarray,
                      r_dir: Tuple | np.ndarray,
                      rect: Tuple[float, float, float, float]):
    """
    ray and rect AABB detect, use slab based method. see:
    https://blog.csdn.net/qq_22822335/article/details/50930423



    Args:
        r_s: 射线起点
        r_dir: 射线方向
        rect:[min_x,min_y,max_x,max_y]

    Returns:

    """
    min_x, min_y, max_x, max_y = rect
    rs_x, rs_y = r_s
    if r_dir[0] == 0:  # 射线平行y轴
        if rs_x < min_x or rs_x > max_x:
            return False
        elif min_y <= rs_y <= max_y:
            return True
        elif rs_y < min_y and r_dir[1] > 0:
            return True
        elif rs_y > max_y and r_dir[1] < 0:
            return True
        else:
            return False
    elif r_dir[1] == 0:  # 射线平行x轴
        if rs_y < min_y or rs_y > max_y:
            return False
        elif min_x <= rs_x <= max_x:
            return True
        elif rs_x < min_x and r_dir[0] > 0:
            return True
        elif rs_x > max_x and r_dir[0] < 0:
            return True
        else:
            return False
    else:  # 不平行于任何一轴
        slab_x_t1 = (min_x - rs_x) / r_dir[0]
        slab_x_t2 = (max_x - rs_x) / r_dir[0]
        slab_y_t1 = (min_y - rs_y) / r_dir[1]
        slab_y_t2 = (max_y - rs_y) / r_dir[1]
        slab_x_tmin = min(slab_x_t1, slab_x_t2)
        slab_x_tmax = max(slab_x_t1, slab_x_t2)
        slab_y_tmin = min(slab_y_t1, slab_y_t2)
        slab_y_tmax = max(slab_y_t1, slab_y_t2)
        # 若射线与x_slab区域的交点缩放比t_xmin和t_xmax均小于1（y_slab区域同理），代表射线背向x或yslab发射，这一定不会相交
        if (slab_x_tmin < 0 and slab_x_tmax < 0) or (slab_y_tmin < 0 and slab_y_tmax < 0):
            return False
        # 两个slab的射线交点缩放比t的范围有交集，一定穿过矩形
        elif slab_x_tmax >= slab_y_tmin or slab_x_tmin >= slab_y_tmax:
            return True
        else:
            all_t = sorted([slab_x_tmin, slab_x_tmax, slab_y_tmin, slab_y_tmax])
            # 排除误差影响，两个交集交于1点时，数学上成立，但python浮点数精度导致两个交点值不一定绝对相等
            if abs(all_t[1] - all_t[2]) < EPS:
                return True
            else:
                return False


# ----------------------------------------------------凸多边形交集相关-----------------------------------------------------

def calc_poly_angle(p, inner_p):
    vec = np.array(p) - inner_p
    ang = np.arctan2(vec[1], vec[0])
    ang = ang if ang >= 0 else ang + PI_2
    return ang


def quick_sort_poly_points(inner_p, poly_points: list, low, high):
    if low >= high:
        return poly_points
    i = low
    j = high
    # 定义基准,基准左边小于基数,右边大于基数
    pivot = poly_points[low]
    ang_piv = calc_poly_angle(pivot, inner_p)
    while i < j:
        # 从后向前扫描
        while i < j and calc_poly_angle(poly_points[j], inner_p) > ang_piv:
            j -= 1
        poly_points[i] = poly_points[j]
        # 从前向后扫描
        while i < j and calc_poly_angle(poly_points[i], inner_p) < ang_piv:
            i += 1
        poly_points[j] = poly_points[i]
    poly_points[j] = pivot
    # 分段排序
    quick_sort_poly_points(inner_p, poly_points, low, j - 1)
    quick_sort_poly_points(inner_p, poly_points, j + 1, high)
    return poly_points


def reorder_conv_vertex(poly_points: list):
    tri_ps = []
    for p in poly_points:
        if len(tri_ps) == 0:
            tri_ps.append(p)
        else:
            can_add = True
            for tp in tri_ps:
                if abs(tp[0] - p[0]) < EPS and abs(tp[1] - p[1]) < EPS:
                    can_add = False
                    break
            if can_add:
                tri_ps.append(p)
            if len(tri_ps) == 3:
                break
    inner_p = calc_poly_centroid(np.array(tri_ps))

    reordered_poly = []
    for p in poly_points:
        if len(reordered_poly) == 0:
            reordered_poly.append(p)
        else:
            r_num = len(reordered_poly)
            insert_i = 0
            for i in range(r_num):
                rp = reordered_poly[i]
                if abs(rp[0] - p[0]) < EPS and abs(rp[1] - p[1]) < EPS:
                    insert_i = None
                    break
                else:
                    insert_i = i
                    vec1, vec2 = np.array(rp) - inner_p, np.array(p) - inner_p
                    ang1, ang2 = np.arctan2(vec1[1], vec1[0]), np.arctan2(vec2[1], vec2[0])
                    ang1 = ang1 if ang1 >= 0 else ang1 + PI_2
                    ang2 = ang2 if ang2 >= 0 else ang2 + PI_2
                    if ang1 > ang2:
                        break
            if insert_i is not None:
                reordered_poly = reordered_poly[0:insert_i] + [p] + reordered_poly[insert_i:r_num + 1]

    return reordered_poly


def cmp_convex_vertex_order(points_set: list):
    """
    对多边形顶点进行排序，排序按照逆时针排序

    Args:
        points_set:

    Returns:

    """
    centroid = calc_poly_centroid(np.array(points_set[0:3]))
    point_num = len(points_set)
    point_angs = []
    reordered_poly = []
    reordered_angs = []
    for p in points_set:
        vec = np.array(p) - centroid
        ang_p = np.arctan2(vec[1], vec[0])
        ang_p = ang_p if ang_p >= 0 else ang_p + PI_2
        point_angs.append(ang_p)

    for i in range(0, point_num - 1):
        for j in range(0, point_num - i - 1):
            ang1 = point_angs[j]
            ang2 = point_angs[j + 1]
            if ang1 > ang2:  # 点p1以多边形质心centroid逆时针绕过的角度大于点p2，这代表p1应该排在p2的后面
                temp = np.array(points_set[j]).tolist()
                points_set[j] = points_set[j + 1]
                points_set[j + 1] = temp
                temp_p = point_angs[j]
                point_angs[j] = point_angs[j + 1]
                point_angs[j + 1] = temp_p
    return points_set


def calc_con_polys_intersect(poly1: list, poly1_in_cir, poly2: list, poly2_in_cir):
    """
    计算两个凸多边形交集，获得最终凸交集轮廓（按照逆时针方向排序）

    Args:
        poly1:
        poly1_in_cir:
        poly2:
        poly2_in_cir:

    Returns:
        交集区域轮廓（凸多边形）: list
    """
    if len(poly1) < 3 or len(poly2) < 3:
        inter_poly = None
    else:
        poly1_num, poly2_num = len(poly1), len(poly2)
        total_set = []
        for p in poly2:
            if chk_p_in_poly(p, poly1):
                total_set.append(p)
        for i in range(-1, poly1_num - 1):  # 先检测poly1的所有顶点在不在poly2中,同时把poly1与poly2边的交点得到
            po1v1, po1v2 = poly1[i], poly1[i + 1]
            if chk_p_in_conv(np.array(po1v2), poly2, poly2_in_cir):
                can_add = True
                for ep in total_set:
                    if abs(ep[0] - po1v2[0]) < EPS and abs(ep[1] - po1v2[1]) < EPS:  # 判断两点是否相同，去重复
                        can_add = False
                        break
                if can_add:
                    total_set.append(po1v2)
            for j in range(-1, poly2_num - 1):
                po2v1, po2v2 = poly2[j], poly2[j + 1]
                # 这里特别注意，由于交点坐标精度问题，可能出现极小的误差，这里需要进一步处理一下，忽略误差
                i_t, i_p, _ = calc_nonparallel_lines_intersect(po1v1, po1v2, po2v1, po2v2)
                # 仅考察poly1和poly2非共线边的交叉点,原因是若两边共线，无非就是重合或完全不相交
                # 若重合，则重合点在之前判断多边形顶点是否在另一多边形内时就会找到
                if i_t:
                    can_add = True
                    for ep in total_set:
                        if abs(ep[0] - i_p[0]) < EPS and abs(ep[1] - i_p[1]) < EPS:  # 判断两点是否相同，去重复
                            can_add = False
                            break
                    if can_add:
                        total_set.append(i_p)

        inter_poly = cmp_convex_vertex_order(total_set) if len(total_set) >= 3 else None

    return inter_poly


def calc_con_polys_intersect_simple(poly1: Sequence, poly2: Sequence):
    """
    计算两个凸多边形交集，获得最终凸交集轮廓（按照逆时针方向排序）

    Args:
        poly1:
        poly1_in_cir:
        poly2:
        poly2_in_cir:

    Returns:
        交集区域轮廓（凸多边形）: list
    """
    if len(poly1) < 3 or len(poly2) < 3:
        inter_poly = None
    else:
        poly1_num, poly2_num = len(poly1), len(poly2)
        total_set = []
        for p in poly2:
            if chk_p_in_conv_simple(p, np.array(poly1)):
                total_set.append(p)
        for i in range(-1, poly1_num - 1):  # 先检测poly1的所有顶点在不在poly2中,同时把poly1与poly2边的交点得到
            po1v1, po1v2 = poly1[i], poly1[i + 1]
            if chk_p_in_conv_simple(po1v2, np.array(poly2)):
                can_add = True
                for ep in total_set:
                    if abs(ep[0] - po1v2[0]) < EPS and abs(ep[1] - po1v2[1]) < EPS:  # 判断两点是否相同，去重复
                        can_add = False
                        break
                if can_add:
                    total_set.append(po1v2)
            for j in range(-1, poly2_num - 1):
                po2v1, po2v2 = poly2[j], poly2[j + 1]
                # 这里特别注意，由于交点坐标精度问题，可能出现极小的误差，这里需要进一步处理一下，忽略误差
                i_t, i_p, _ = calc_nonparallel_lines_intersect(po1v1, po1v2, po2v1, po2v2)
                # 仅考察poly1和poly2非共线边的交叉点,原因是若两边共线，无非就是重合或完全不相交
                # 若重合，则重合点在之前判断多边形顶点是否在另一多边形内时就会找到
                if i_t > 0:
                    can_add = True
                    for ep in total_set:
                        if abs(ep[0] - i_p[0]) < EPS and abs(ep[1] - i_p[1]) < EPS:  # 判断两点是否相同，去重复
                            can_add = False
                            break
                    if can_add:
                        total_set.append(i_p)

        inter_poly = cmp_convex_vertex_order(total_set) if len(total_set) >= 3 else None

    return inter_poly


# ----------------------------------------------------求凸包相关----------------------------------------------------------
def calc_sort_point_cos(points, center_point):
    """
    按照与中心点的极角进行排序，使用的是余弦的方法

    Args:
        points: 需要排序的点
        center_point: 中心点

    Returns:

    """
    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i]
        point = [point_[0] - center_point[0], point_[1] - center_point[1]]
        rank.append(i)
        norm_value = np.sqrt(point[0] * point[0] + point[1] * point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)

    for i in range(0, n - 1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index - 1] or (
                    cos_value[index] == cos_value[index - 1] and norm_list[index] > norm_list[index - 1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index - 1]
                rank[index] = rank[index - 1]
                norm_list[index] = norm_list[index - 1]
                cos_value[index - 1] = temp
                rank[index - 1] = temp_rank
                norm_list[index - 1] = temp_norm
                index = index - 1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])

    return sorted_points


def calc_convex_graham_scan(points):
    """
    Graham扫描法计算凸包

    Args:
        points:

    Returns:

    """
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (
                points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i

    bottom_point = points.pop(min_index)
    sorted_points = calc_sort_point_cos(points, bottom_point)

    m = len(sorted_points)
    if m < 2:
        print("点的数量过少，无法构成凸包")
        return

    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])

    for i in range(2, m):
        length = len(stack)
        top = stack[length - 1]
        next_top = stack[length - 2]
        v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
        v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        while alg.cross(v1, v2) >= 0:
            stack.pop()
            length = len(stack)
            top = stack[length - 1]
            next_top = stack[length - 2]
            v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        stack.append(sorted_points[i])

    return stack


# ----------------------------------------------------三角剖分相关--------------------------------------------------------


def calc_poly_triangulation(poly_bounds: list):
    total_bound, total_edges = calc_poly_out_bound(poly_bounds)
    tris = calc_out_bound_triangulation(total_bound, total_edges)
    return tris


def calc_poly_out_bound(poly_bounds: list):
    """
    将内部轮廓依次融合到最外层轮廓，使得最终轮廓只有一个
    Args:
        poly_bounds:

    Returns:

    """
    target_bounds = pickle.loads(pickle.dumps(poly_bounds))
    if target_bounds is None:
        return None, None
    elif len(target_bounds) == 1:  # 仅包含外轮廓
        total_bound = target_bounds[0]
        ver_num = len(total_bound)
        total_bound, _ = calc_adjust_poly_order(total_bound)
        total_edges = []
        for i in range(-1, ver_num - 1):
            total_edges.append([total_bound[i], total_bound[i + 1], 1])  # 创建轮廓边，最后一位表示是否外轮廓，用1,0表示
    else:
        bounds_num = len(target_bounds)
        out_bound = pickle.loads(pickle.dumps(target_bounds[0]))
        out_bound, _ = calc_adjust_poly_order(out_bound)  # 外轮廓调为逆时针
        added_edges = []
        inner_bounds = pickle.loads(pickle.dumps(target_bounds[1:bounds_num]))
        while len(inner_bounds) > 0:
            selected_idx = 0
            max_x = 0
            for i in range(len(inner_bounds)):
                temp_bound = inner_bounds[i]
                max_v_x = 0
                for iv in temp_bound:
                    if iv[0] > max_v_x:
                        max_v_x = iv[0]
                if max_v_x > max_x:
                    selected_idx = i
                    max_x = max_v_x
            inner_bound = inner_bounds[selected_idx]
            inner_bound, _ = calc_adjust_poly_order(inner_bound, order=0)  # 内轮廓调为顺时针
            in_num = len(inner_bound)
            in_idx, out_idx = find_visible_vertex(inner_bound, out_bound)
            out_num = len(out_bound)
            out1 = out_bound[0:out_idx + 1]
            out2 = out_bound[out_idx:out_num]
            in1 = inner_bound[in_idx:in_num]
            in2 = inner_bound[0:in_idx + 1]
            out_bound = out1 + in1 + in2 + out2
            added_edges.append([out_bound[out_idx], inner_bound[in_idx]])
            added_edges.append([inner_bound[in_idx], out_bound[out_idx]])
            inner_bounds.pop(selected_idx)

        total_bound = out_bound
        ver_num = len(total_bound)
        total_edges = []
        for i in range(-1, ver_num - 1):
            is_out_e = 1
            for added_e in added_edges:
                if chk_dir_edge_same([total_bound[i], total_bound[i + 1]], added_e):
                    is_out_e = 0
                    break
            total_edges.append([total_bound[i], total_bound[i + 1], is_out_e])  # 创建轮廓边，最后一位表示是否外轮廓，用1,0表示

    return total_bound, total_edges


def calc_out_bound_triangulation(poly_bounds: list, poly_edges: list):
    """
    calculate the triangulation of a polygon, for the outer contour of polygon, the order of vertex must be
    anti-clockwise, and the inner contours are all in clockwise order.

    Args:
        poly_bounds: a list contains all bounds, [0] must be outer bound.

        poly_edges:

    Returns:
        list of triangulation
    """

    total_bound = pickle.loads(pickle.dumps(poly_bounds))
    total_edges = pickle.loads(pickle.dumps(poly_edges))
    total_num = len(total_bound)
    if total_num < 3:
        return None
    else:
        tris = []
        total_count = len(total_bound)
        count = 0
        while len(total_bound) > 3:
            count += 1
            total_bound, total_edges, tri = ear_clip_poly_opti(total_bound, total_edges)  # or use 'ear_clip_poly'
            tris.append(tri)
            print("\rtriangulation:{:.2f}%".format(count / total_count * 100), end="")
        print()
        tri = Triangle()
        tri.set_points(total_bound[0], total_bound[1], total_bound[2])
        tri.vertices = cmp_convex_vertex_order(tri.vertices)
        tri.poly_contour = Polygon(tri.vertices)
        tri.in_circle = calc_poly_max_in_circle(tri.vertices)
        e1 = remove_edge_from_edgeset(total_edges, [total_bound[0], total_bound[1]])
        e2 = remove_edge_from_edgeset(total_edges, [total_bound[1], total_bound[2]])
        e3 = remove_edge_from_edgeset(total_edges, [total_bound[0], total_bound[2]])
        if e1[2]:
            tri.out_edges.append([tri.find_vertex_idx(e1[0]), tri.find_vertex_idx(e1[1])])
        else:
            tri.in_edges.append([tri.find_vertex_idx(e1[0]), tri.find_vertex_idx(e1[1])])
        if e2[2]:
            tri.out_edges.append([tri.find_vertex_idx(e2[0]), tri.find_vertex_idx(e2[1])])
        else:
            tri.in_edges.append([tri.find_vertex_idx(e2[0]), tri.find_vertex_idx(e2[1])])
        if e3[2]:
            tri.out_edges.append([tri.find_vertex_idx(e3[0]), tri.find_vertex_idx(e3[1])])
        else:
            tri.in_edges.append([tri.find_vertex_idx(e3[0]), tri.find_vertex_idx(e3[1])])
        tris.append(tri)
        return tris


def chk_poly_clockwise(poly_bound: list):
    """
    检查多边形顺逆时针方向

    Args:
        poly_bound:

    Returns:
        1-逆时针 0-顺时针
    """
    s = 0
    poly_num = len(poly_bound)
    for i in range(-1, poly_num - 1):
        s += (poly_bound[i + 1][1] + poly_bound[i][1]) * (poly_bound[i][0] - poly_bound[i + 1][0])
    ori_order = 1 if s > 0 else 0
    return ori_order


def calc_adjust_poly_order(poly_bound: list, order=1):
    """
    将多边形轮廓点按照指定方向重新调整，改变为顺时针/逆时针，默认逆时针. 1. 先利用鞋带公式求原始轮廓的方向. 2. 调整为目标方向

    Args:
        poly_bound: 轮廓点

        order: 1-逆时针，0-顺时针

    Returns:
        调整后的轮廓点
    """
    adjusted = pickle.loads(pickle.dumps(poly_bound))
    ori_order = chk_poly_clockwise(adjusted)
    if (order and not ori_order) or (not order and ori_order):
        adjusted.reverse()
    return adjusted, ori_order


def ear_clip_poly(bound, edges):
    """
    从给定轮廓中用耳切法切掉一个耳朵

    Args:
        bound:
        edges:

    Returns:

    """
    cut_ear = None
    bound_num = len(bound)

    for i in range(0, bound_num):
        is_ear = True
        v_f = bound[i - 1]  # vn-1
        v = bound[i]  # vn
        v_n = bound[(i + 1) % bound_num]  # vn+1
        vec1 = np.array([v[0] - v_f[0], v[1] - v_f[1]])
        vec2 = np.array([v_n[0] - v[0], v_n[1] - v[1]])
        if calc_angle_bet_vec(vec2, vec1) < 0:  # 当前拐角不是凸的
            is_ear = False
        else:  # 当前拐角是凸的，可能是耳朵
            tri_c = np.array([(v_f[0] + v[0] + v_n[0]) / 3, (v_f[1] + v[1] + v_n[1]) / 3])  # 当前三个点构成三角形的重心
            for j in range(0, bound_num - 3):
                other_v = np.array(bound[(i + j + 2) % bound_num])  # 除刚才三个点外的其他点
                in_tri = False
                nv_f = np.array(v_f)
                nv = np.array(v)
                nv_n = np.array(v_n)
                if chk_ps_on_line_side(tri_c, other_v, nv_f, nv):
                    if chk_ps_on_line_side(tri_c, other_v, nv, nv_n):
                        if chk_ps_on_line_side(tri_c, other_v, nv_n, nv_f):
                            in_tri = True
                if in_tri:  # 存在轮廓点在当前您的三角中，代表这个不是耳朵
                    is_ear = False
                    break
        if is_ear:
            bound.pop(i)
            e1 = remove_edge_from_edgeset(edges, [v_f, v], is_dir=1)
            e2 = remove_edge_from_edgeset(edges, [v, v_n], is_dir=1)
            e3 = [v_f, v_n, 0]  # 创建地边一定是内部边
            edges.append(e3)
            cut_ear = Triangle()
            cut_ear.set_points(v_f, v, v_n)
            cut_ear.vertices = cmp_convex_vertex_order(cut_ear.vertices)
            cut_ear.in_circle = calc_poly_max_in_circle(cut_ear.vertices)
            if e1[2]:
                cut_ear.out_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
            else:
                cut_ear.in_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
            if e2[2]:
                cut_ear.out_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
            else:
                cut_ear.in_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
            cut_ear.in_edges.append([cut_ear.find_vertex_idx(e3[0]), cut_ear.find_vertex_idx(e3[1])])
            break

    return bound, edges, cut_ear


def ear_clip_poly_opti(bound, edges):
    """
        从给定轮廓中用耳切法切掉一个耳朵

        Args:
            bound:
            edges:

        Returns:

        """
    cut_ear = None
    bound_num = len(bound)
    ears = []
    for i in range(0, bound_num):
        is_ear = True
        v_f = bound[i - 1]  # vn-1
        v = bound[i]  # vn
        v_n = bound[(i + 1) % bound_num]  # vn+1
        nv_f = np.array(v_f)
        nv = np.array(v)
        nv_n = np.array(v_n)
        vec1 = np.array([v[0] - v_f[0], v[1] - v_f[1]])
        vec2 = np.array([v_n[0] - v[0], v_n[1] - v[1]])
        if calc_angle_bet_vec(vec2, vec1) <= 0:  # 当前拐角不是凸的
            is_ear = False
        else:  # 当前拐角是凸的，可能是耳朵
            tri_c = np.array([(v_f[0] + v[0] + v_n[0]) / 3, (v_f[1] + v[1] + v_n[1]) / 3])  # 当前三个点构成三角形的重心
            for j in range(0, bound_num - 3):
                other_v = np.array(bound[(i + j + 2) % bound_num])  # 除刚才三个点外的其他点
                in_tri = False
                if chk_ps_on_line_side(tri_c, other_v, nv_f, nv):
                    if chk_ps_on_line_side(tri_c, other_v, nv, nv_n):
                        if chk_ps_on_line_side(tri_c, other_v, nv_n, nv_f):
                            in_tri = True
                if in_tri:  # 存在轮廓点在当前您的三角中，代表这个不是耳朵
                    is_ear = False
                    break
        if is_ear:
            ears.append(i)
    target_idx = 0
    tar_vf, tar_v, tar_vn = None, None, None
    max_area = 0  # 选面积最大的耳朵
    for idx in ears:
        v_f = bound[idx - 1]  # vn-1
        v = bound[idx]  # vn
        v_n = bound[(idx + 1) % bound_num]  # vn+1

        area = calc_poly_area([v_f, v, v_n])
        if area > max_area:
            target_idx = idx
            tar_vf, tar_v, tar_vn = v_f, v, v_n
            max_area = area
    bound.pop(target_idx)
    e1 = remove_edge_from_edgeset(edges, [tar_vf, tar_v], is_dir=1)
    e2 = remove_edge_from_edgeset(edges, [tar_v, tar_vn], is_dir=1)
    e3 = [tar_vf, tar_vn, 0]  # 创建地边一定是内部边
    edges.append(e3)
    cut_ear = Triangle()
    cut_ear.set_points(tar_vf, tar_v, tar_vn)
    cut_ear.vertices = cmp_convex_vertex_order(cut_ear.vertices)
    cut_ear.poly_contour = Polygon(cut_ear.vertices)
    cut_ear.in_circle = calc_poly_max_in_circle(cut_ear.vertices)
    if e1[2]:
        cut_ear.out_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
    else:
        cut_ear.in_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
    if e2[2]:
        cut_ear.out_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
    else:
        cut_ear.in_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
    cut_ear.in_edges.append([cut_ear.find_vertex_idx(e3[0]), cut_ear.find_vertex_idx(e3[1])])

    return bound, edges, cut_ear


def find_visible_vertex(inner_bound, outer_bound):
    """
    返回指定内部轮廓与外部轮廓间的一对相互可见顶点，返回值是相应顶点在各自轮廓中的index

    Args:
        inner_bound:
        outer_bound:

    Returns:
        内轮廓可见点idx，外轮廓可见点idx
    """
    M = None
    m_idx = 0
    mx_max = 0
    in_num = len(inner_bound)
    for j in range(0, in_num):
        iv = inner_bound[j]
        if iv[0] >= mx_max:
            M = iv
            m_idx = j
            mx_max = iv[0]
    out_num = len(outer_bound)
    intersect_t = float('inf')
    I, cor_b1, cor_b2 = None, None, None
    cor_b1_idx, cor_b2_idx = 0, 0
    for i in range(-1, out_num - 1):
        ov1 = outer_bound[i]
        ov2 = outer_bound[i + 1]
        i_p, i_t = find_ray_edge_intersect(M, ov1, ov2)
        if i_p is not None:  # 有交点
            if 0 <= i_t <= intersect_t:
                I, cor_b1, cor_b2 = i_p, ov1, ov2
                cor_b1_idx, cor_b2_idx = i, i + 1
                intersect_t = i_t
    if chk_p_same(I, cor_b1):  # 如果交点就是外轮廓上的一个点
        return m_idx, cor_b1_idx
    elif chk_p_same(I, cor_b2):  # 如果交点就是外轮廓上的一个点
        return m_idx, cor_b2_idx
    else:  # 焦点在外轮廓的一条边上，需要进一步处理
        if cor_b1[0] > cor_b2[0]:
            P = cor_b1
            p_idx = cor_b1_idx
        else:
            P = cor_b2
            p_idx = cor_b2_idx
        p_in_MIP = []
        tri_c = np.array([(M[0] + I[0] + P[0]) / 3, (M[1] + I[1] + P[1]) / 3])  # 当前三个点构成三角形的重心
        nM, nI, nP = np.array(M), np.array(I), np.array(P)
        for r_i in range(0, out_num - 2):
            idx = (cor_b2_idx + r_i + 1) % out_num
            other_v = np.array(outer_bound[idx])  # 除刚才三个点外的其他点
            if chk_ps_on_line_side(tri_c, other_v, nM, nI):
                if chk_ps_on_line_side(tri_c, other_v, nI, nP):
                    if chk_ps_on_line_side(tri_c, other_v, nP, nM):
                        p_in_MIP.append(idx)
        if len(p_in_MIP) == 0:  # 如果没有别的外轮廓点在三角形MIP中，则MP构成一对相互可见点
            return m_idx, p_idx
        else:  # 存在，则从中找出与x轴夹角最小的点与M构成一对相互可见点
            min_a = float('inf')
            for p_i in p_in_MIP:
                potential_p = outer_bound[p_i]
                v_mp = np.array(potential_p) - np.array(M)
                v_len = alg.l2_norm(v_mp)
                if v_len < min_a:
                    p_idx = p_i
                    min_a = v_len
            return m_idx, p_idx


def find_ray_edge_intersect(m, p1, p2):
    """
    以m为基，向右x方向发射射线，计算射线与线段p1p2的交点

    Args:
        m:
        p1:
        p2:

    Returns:

    """
    if p1[1] > m[1] and p2[1] > m[1]:  # 线段在射线上面
        i_p = None
        t = None
    elif p1[1] < m[1] and p2[1] < m[1]:  # 线段在射线下面
        i_p = None
        t = None
    elif p1[1] == m[1] and p2[1] != m[1]:  # 线段一头在射线上，一头不在
        i_p = p1
        t = p1[0] - m[0]
    elif p2[1] == m[1] and p1[1] != m[1]:  # 线段一头在射线上，一头不在
        i_p = p2
        t = p2[0] - m[0]
    elif p1[1] == m[1] and p2[1] == m[1]:  # 线段完全在射线上
        d1 = p1[0] - m[0]
        d2 = p2[0] - m[0]
        if abs(d1) > abs(d2):  # 选距离近的点作为交点
            i_p = p2
            t = d2
        else:
            i_p = p1
            t = d1
    else:
        v = np.array(p2) - np.array(p1)
        i_p = (np.array(p1) + np.multiply(v, abs(m[1] - p1[1]) / abs(p2[1] - p1[1]))).tolist()
        t = i_p[0] - m[0]
    return i_p, t


def remove_edge_from_edgeset(edge_set, tar, is_dir=0):
    """
    从指定边集中检测并删除目标边，若有则返回删除边，若无则返回空. 默认针对无向边

    Args:
        edge_set:
        tar:
        is_dir:

    Returns:

    """
    if tar is None:
        return None
    elif not is_dir:
        for e in edge_set:
            if chk_edge_same(e, tar):
                edge_set.remove(e)
                return e
        return None
    else:
        for e in edge_set:
            if chk_dir_edge_same(e, tar):
                edge_set.remove(e)
                return e
        return None


def chk_edge_same(e1, e2):
    """
    判断无向边是否相同

    Args:
        e1:
        e2:

    Returns:

    """
    if chk_p_same(e1[0], e2[0]) and chk_p_same(e1[1], e2[1]):
        return True
    elif chk_p_same(e1[0], e2[1]) and chk_p_same(e1[1], e2[0]):
        return True
    else:
        return False


def chk_dir_edge_same(e1, e2):
    """
    判断无向边是否相同

    Args:
        e1:
        e2:

    Returns:

    """
    if chk_p_same(e1[0], e2[0]) and chk_p_same(e1[1], e2[1]):
        return True
    else:
        return False


def chk_p_same(p1, p2):
    return abs(p1[0] - p2[0]) < EPS and abs(p1[1] - p2[1]) < EPS


def chk_is_triangle(tri_contour):
    p1 = np.array(tri_contour[0])
    p2 = np.array(tri_contour[1])
    p3 = np.array(tri_contour[2])
    v1 = p1 - p2
    v2 = p3 - p2

    if alg.cross(v1, v2) == 0:
        return False
    else:
        return True


def obtain_hex_points(w):
    """
    生成正六边形顶点

    Args:
        w: 边长

    Returns:
        六边形顶点集
    """
    hex_points = [[0.5 * w, 0],
                  [1.5 * w, 0],
                  [2 * w, 0.5 * 3 ** 0.5 * w],
                  [1.5 * w, 3 ** 0.5 * w],
                  [0.5 * w, 3 ** 0.5 * w],
                  [0, 0.5 * 3 ** 0.5 * w]]
    return hex_points


def obtain_pent_points(w):
    """
    生成正五边形顶点

    Args:
        w: 边长

    Returns:
        五边形顶点集
    """
    pent_angle = 0.4 * PI
    cos_a = np.cos(pent_angle)
    sin_a = np.sin(pent_angle)
    pent = [[cos_a * w, 0],
            [(1 + cos_a) * w, 0],
            [(1 + 2 * cos_a) * w, sin_a * w],
            [(0.5 + cos_a) * w, np.tan(pent_angle) * 0.5 * w],
            [0, sin_a * w]]
    return pent


def obtain_rect_points(w, h):
    """
    生成矩形
    Args:
        w: 宽
        h: 高

    Returns:
        矩形顶点集
    """
    x_min = 0
    x_max = w
    y_min = 0
    y_max = h
    rect = [[x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]]
    return rect
