import bisect
import random

import numpy as np
from shapely import GeometryCollection

from pyrdw.core.space import *
from pyrdw.core.space.visibility import SimpleRay


class Tiling:

    def __init__(self):
        self.id: int = 0
        self.type: int = 1  # 1 totally inside space, 0 - some or all part collides with space
        self.corr_scene = None
        self.mat_loc: Tuple = tuple()  # row, col
        self.center: np.ndarray = np.array([0, 0])
        self.cross_bound: np.ndarray = np.array([])  # 穿过这个tiling的边界
        self.cross_area = np.array([])  # tiling 与场景相交的区域，仅当type=0时记录
        self.rect: np.ndarray = np.array([])
        self.poly_contour: Polygon = Polygon()
        self.nei_ids: np.ndarray = np.array([])
        self.rela_patch = None
        self.corr_conv_ids = []  # 相交的凸多边形区域id
        self.corr_conv_cin = -1  # tiling center 所在的凸多边形区域id

        self.vis_multi_areas = False
        self.vis_tri: Tuple[np.ndarray] = tuple()
        self.vis_poly: Tuple[Polygon] = tuple()
        self.vis_rays: Tuple[Tuple] = tuple()
        # 以[0,1]旋转角度划分若干区域，每个区域内的可达tiling，目前将角度划分为120个区间，每个区间代表3度（见space对象）
        self.vis_grids = tuple()  # 周围360度可见区域离散网格，1度为一个离散扇区，每个扇区记录网格id
        self.nearst_obs_pos = np.array([])

        self.nearst_obs_gid = 0  # 距离最近障碍物tiling的id（此处采用occupancy grid方法，求与最近障碍物tiling的距离）
        self.sur_gids: Tuple = tuple()  # 指定半径范围内的其他tiling的ids,指定半径使用scene对象的human_step_single数值
        self.sur_obs_gids = []  # 指定半径范围内的障碍物tiling的ids
        self.sur_bound_gids = []  # 指定半径范围内的障碍物边界tiling的ids（1个human step 范围内）
        self.sur_occu_safe = True  # 周围1步内无遮挡则为true

    def calc_rays_vis_poly_intersects(self, pos: np.ndarray, fwds: np.ndarray):
        """

        Args:
            pos:
            fwds:

        Returns:

        """
        p_len = len(self.vis_rays)
        result = []
        for fid in range(fwds.shape[0]):
            fwd = fwds[fid]
            tr = SimpleRay()
            tr.origin = self.center
            tr.hit = self.center + fwd * 1e7
            tr.rot_angle = geo.calc_axis_x_angle(fwd)
            tr_rid1 = bisect.bisect_left(self.vis_rays, tr.rot_angle, key=lambda x: x.rot_angle)
            if tr_rid1 == 0:
                tr_rid2 = -1
            elif tr_rid1 == p_len:
                tr_rid2 = 0
                tr_rid1 = -1
            else:
                tr_rid2 = tr_rid1 - 1
            vis_line = [self.vis_rays[tr_rid1].hit, self.vis_rays[tr_rid2].hit]
            vis_inter = np.array(LineString([tr.origin, tr.hit]).intersection(LineString(vis_line)).coords)
            if vis_inter.shape[0] == 0:
                print(np.array(vis_line), np.array([tr.origin, tr.hit]))
            else:
                vis_inter = np.array(LineString([tr.origin, tr.hit]).intersection(LineString(vis_line)).coords)[0]
            vis_dis = alg.l2_norm(vis_inter - pos)
            result.append((vis_inter, vis_dis, vis_line))
        return result

    def intersection_scene(self, scene):
        scene_poly: Polygon = scene.poly_contour
        tiling_poly: Polygon = self.poly_contour
        inter = scene_poly.boundary.intersection(tiling_poly.boundary)
        if inter.is_empty or isinstance(inter, Point):
            if scene_poly.contains(tiling_poly):
                self.type = 1
            else:
                self.type = 0
        else:  # tiling must collide with boundary
            self.type = 0
            self.cross_area = []
            cross_geo = scene_poly.intersection(tiling_poly)
            if isinstance(cross_geo, LineString):
                return
            elif isinstance(cross_geo, Polygon):
                self.cross_area.append(np.array(cross_geo.exterior.coords))
            elif isinstance(cross_geo, MultiPolygon):
                for geom in cross_geo.geoms:
                    self.cross_area.append(np.array(geom.exterior.coords))
            elif isinstance(cross_geo, GeometryCollection):
                for geom in cross_geo.geoms:
                    if isinstance(geom, Polygon):
                        self.cross_area.append(np.array(geom.exterior.coords))

            bound_lines: List[LineString] = []
            if isinstance(scene_poly.boundary, LineString):
                bound_lines.append(scene_poly.boundary)
            else:
                for line in scene_poly.boundary.geoms:
                    bound_lines.append(line)
            cross_bound = []
            for l in bound_lines:
                data = np.array(l.coords)
                num, _ = data.shape
                for i in range(num - 1):
                    s = data[i]
                    e = data[i + 1]
                    inter = self.poly_contour.intersection(LineString([s, e]))
                    if not inter.is_empty and isinstance(inter, LineString):
                        cross_bound.append(np.array([s, e]))
            self.cross_bound = np.array(cross_bound)

    def obtain_vis_attr_id(self, pos):
        if not self.vis_multi_areas:
            return 0
        else:
            for cid, c_area in enumerate(self.cross_area):
                c_poly = Polygon(c_area)
                if c_poly.covers(Point(pos)):
                    return cid
            else:
                print('grid multi vis error', pos, self.cross_area)  # 主要问题是pos可能落在tiling的空白区域
                return np.random.randint(len(self.cross_area))
