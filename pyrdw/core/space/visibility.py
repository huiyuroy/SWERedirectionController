from pyrdw.core.space import *


class SimpleRay:
    def __init__(self):
        self.origin = (0, 0)
        self.hit = None
        self.rot_angle = 0


class Ray(SimpleRay):
    def __init__(self):
        super().__init__()
        self.origin = (0, 0)
        self.dir_point = (0, 0)
        self.fwd = np.array((0, 0))
        self.hit_dis = 0
        self.rela_vertex: Vertex = None
        self.pot_coll_walls: List[Segment] = []  # 潜在碰撞墙
        self.exceed_vertex = False  # 是否超越轮廓顶点，延伸向远方
        self.excluded = False


class Vertex:
    def __init__(self, data):
        self.data: Tuple = data
        self.rela_data: np.ndarray = np.array(data)  # 与当前观察点的相对位置，每次更新可见性时更新
        self.rot_angle: float = 0  # 以观察点为原点时，绕x轴转过的角度
        self.quadrant: int = 0  # 0,1,2,3,4 象限
        self.dis2obs_pos = 0
        self.start_seg: Segment = Segment()  # 以vertex为起点的segment
        self.end_seg: Segment = Segment()  # 以vertex为终点的segment
        self.excluded = False


class Segment:
    def __init__(self):
        self.id: int = 0
        self.vtx_start: Vertex = None
        self.vtx_end: Vertex = None
        self.line: LineString = None
        self.pre: Segment = None
        self.post: Segment = None
        self.length: float = 0
        self.center: Tuple = (0, 0)
        self.out_contour = True
        # self.bound = np.zeros((2, 2))  # [[min_x,min_y],[max_x,max_y]]
        self.aabb = (0, 0, 0, 0)
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0
        self.dis2obs_pos = 0
        self.lighted = False


class Observer:
    """
    Realized by Rotational sweep visibility. see:
    https://www.redblobgames.com/articles/visibility/
    https://github.com/akapkotel/light_raycasting

    Idea: Observer is a point which represents a source of light or an observer in field-of-view simulation.
    """

    def __init__(self):
        self.pos = (0, 0)  # position of the light/observer
        self.scene = None
        # our algorithm does not check against whole polygons-obstacles, but against each of their edges:
        # objects considered as blocking FOV/light. Each obstacle is a polygon consisting a list of points -
        # it's vertices.
        self.fov_fwd = (0, 0)
        self.fov_angle = 0
        self.half_fov = 85  # degree

        self.segments: List[Segment] = []
        self.vertexes: List[Vertex] = []
        self.outer_segments: List[Segment] = []
        self.inner_segments: List[Segment] = []
        self.face_segments: Set[Segment] = set()  # 面向光源的墙壁
        self.rays: Sequence[Ray] = []
        # this would be used to draw our visible/lit-up area:

    def load_walls(self):
        self.vertexes = []
        self.segments = []
        self.outer_segments = []
        self.inner_segments = []
        seg_id = 0
        for bound in self.scene.bounds:
            if bound.is_out_bound:
                b_points, _ = geo.calc_adjust_poly_order(bound.points, order=0)  # 必须全部方向调整为顺时针
            else:
                b_points, _ = geo.calc_adjust_poly_order(bound.points, order=1)  # 必须全部方向调整为逆时针
            vertexes = [Vertex(p) for p in b_points]
            segments = []

            for i in range(bound.points_num):
                s_id = i
                e_id = i + 1 if i < bound.points_num - 1 else 0
                wall = Segment()
                wall.id = seg_id
                wall_s, wall_e = tuple(b_points[s_id]), tuple(b_points[e_id])
                wall.vtx_start = vertexes[s_id]
                wall.vtx_end = vertexes[e_id]
                wall.line = LineString([wall.vtx_start.data, wall.vtx_end.data])
                wall.vtx_start.start_seg = wall
                wall.vtx_end.end_seg = wall
                wall.out_contour = bound.is_out_bound
                wall.length = alg.l2_norm(np.array(wall_s) - np.array(wall_e))
                wall.center = tuple((np.array(wall_s) + np.array(wall_e)) * 0.5)

                wall.min_x = min(wall_s[0], wall_e[0])
                wall.min_y = min(wall_s[1], wall_e[1])
                wall.max_x = max(wall_s[0], wall_e[0])
                wall.max_y = max(wall_s[1], wall_e[1])
                wall.aabb = (wall.min_x, wall.min_y, wall.max_x, wall.max_y)
                if bound.is_out_bound:
                    self.outer_segments.append(wall)
                else:
                    self.inner_segments.append(wall)
                segments.append(wall)
                seg_id += 1
            self.vertexes.extend(vertexes)
            self.segments.extend(segments)

        for wall in self.segments:
            w_s = wall.vtx_start
            wall.pre = w_s.end_seg
            w_s.end_seg.post = wall

    def update_visible_polygon(self, pos) -> Sequence[Ray]:
        """
        Field of view or lit area is represented by polygon which is basically a list of points. Each frame list is
        updated accordingly to the position of the Observer

        Args:
            pos: point from which we will shot rays

        Returns:
            visible rays: sorted by rotation angle
        """
        self.pos = pos
        self.__update_entire_relative_relation()
        vertexes = self.vertexes[::]  # [v for v in self.vertexes]
        vertexes.sort(key=lambda v: v.rot_angle)
        walls = self.segments[::]  # [s for s in self.segments]
        # sorting walls according to their distance to origin allows for faster finding rays intersections and avoiding
        # iterating through whole list of the walls:
        for w in walls:
            w_s, w_e = w.vtx_start.data, w.vtx_end.data
            w.dis2obs_pos = geo.calc_point_mindis2line(self.pos, w_s, w_e)
        walls.sort(key=lambda w: w.dis2obs_pos)
        # to avoid issue with border-walls when wall-rays are preceding obstacle-rays:
        walls.sort(key=lambda w: w.out_contour)
        # s = time.perf_counter()
        self.__generate_rays_for_walls()
        self.__intersect_rays_w_walls()
        # e = time.perf_counter()
        # print('obtain vis polys in {}, fps {}'.format(e - s, 1 / (e - s)))
        # need to sort rays by their ending angle again because offset_rays are unsorted and pushed at the end of the
        # list: finally, we build a visibility polygon using endpoint of each ray:

        return self.rays

    def __update_entire_relative_relation(self):
        n_pos = np.array(self.pos)
        for vertex in self.vertexes:
            dis_vec = np.array(vertex.data) - n_pos
            vertex.rela_data = tuple(dis_vec)
            vertex.rot_angle = geo.calc_axis_x_angle(vertex.rela_data)
            vertex.dis2obs_pos = alg.l2_norm(dis_vec)
            vertex.excluded = False

    def __generate_rays_for_walls(self):
        """
        Create a 'ray' connecting origin with each corner (obstacle vertex) on the screen. Ray is a tuple of two (x, y)
        coordinates used later to find which segment obstructs visibility.
        TODO: find way to emit less offset rays [x][ ]
        :param origin: Tuple -- point from which 'light' is emitted
        :param corners: List -- vertices of obstacles
        :return: List -- rays to be tested against obstacles edges

        Args:
            vertexes:

        """
        self.rays = set()
        self.face_segments = set()
        for wall in self.outer_segments:
            wall.lighted = False
            w_s, w_e = wall.vtx_start, wall.vtx_end
            if geo.chk_points_clockwise((self.pos, w_s.data, w_e.data)) >= 0:  # 如果顺时针或共线，代表这面外墙正面朝着光源
                self.face_segments.add(wall)
                wall.lighted = True
        for wall in self.inner_segments:
            wall.lighted = False
            w_s, w_e = wall.vtx_start, wall.vtx_end
            if not geo.chk_points_clockwise((self.pos, w_s.data, w_e.data)) <= 0:  # 如果逆时针或共线，代表这面内墙正面朝着光源
                self.face_segments.add(wall)
                wall.lighted = True

        dropout_segs = []
        for face_seg in self.face_segments:
            ws, we = face_seg.vtx_start.data, face_seg.vtx_end.data
            for other_seg in self.face_segments:
                if face_seg.id != other_seg.id:  # 另一个面向光源的墙
                    o_ws, o_we = other_seg.vtx_start.data, other_seg.vtx_end.data
                    p2s, p2e, s_l = LineString([self.pos, ws]), LineString([self.pos, we]), LineString([o_ws, o_we])
                    if p2s.intersects(s_l) and p2e.intersects(s_l):
                        dropout_segs.append(face_seg)
                        break
        for seg in dropout_segs:
            self.face_segments.remove(seg)
        for wall in self.face_segments:
            w_s, w_e = wall.vtx_start, wall.vtx_end
            if not w_s.excluded:
                w_s.excluded = True
                self.rays.add(self.__generate_vertex_ray(w_s))
            if not w_e.excluded:
                w_e.excluded = True
                self.rays.add(self.__generate_vertex_ray(w_e))

        # for wall in self.face_segments:
        extend_rays = []
        for ray in self.rays:
            wall1 = ray.rela_vertex.start_seg
            wall2 = ray.rela_vertex.end_seg
            if wall1.lighted != wall2.lighted:
                r2v1 = self.__generate_extend_ray(ray.rot_angle + EPS)
                r2v2 = self.__generate_extend_ray(ray.rot_angle - EPS)
                extend_rays.append(r2v1)
                extend_rays.append(r2v2)
        self.rays = list(self.rays)
        self.rays.extend(extend_rays)
        self.rays = list({ray.rot_angle: ray for ray in self.rays}.values())
        # sorted by rotation angle between x-axis, which means rays sorted by counterclockwise
        self.rays.sort(key=lambda r: r.rot_angle)

    def __intersect_rays_w_walls(self):
        """
        Check correct vis ray (already sorted).

        Returns:

        """
        for ray in self.rays:
            r_s, r_e = np.array(ray.origin), np.array(ray.dir_point)
            for wall in self.face_segments:
                if not geo.chk_ray_rect_AABB(ray.origin, ray.fwd, wall.aabb):
                    continue
                w_s, w_e = wall.vtx_start, wall.vtx_end
                inter = geo.calc_ray_line_intersection(np.array(w_s.data), np.array(w_e.data), r_s, ray.fwd)
                if inter is not None:
                    if ray.hit is None:
                        ray.hit = inter[-1]
                        ray.hit_dis = alg.l2_norm(ray.hit - np.array(ray.origin))
                    else:
                        ray.pot_coll_walls.append(wall)
                        hit_d = ray.hit_dis
                        end_d = alg.l2_norm(inter[-1] - np.array(ray.origin))
                        if end_d < hit_d:
                            ray.hit = inter[-1]
                            ray.hit_dis = end_d

        self.rays = [r for r in self.rays if r.hit is not None]

    def __generate_vertex_ray(self, vertex: Vertex):
        r2v = Ray()
        r2v.origin = self.pos
        r2v.dir_point = vertex.data
        r2v.fwd = geo.norm_vec((r2v.dir_point[0] - r2v.origin[0], r2v.dir_point[1] - r2v.origin[1]))
        r2v.rot_angle = geo.calc_axis_x_angle(np.array(r2v.dir_point) - np.array(self.pos))
        r2v.rela_vertex = vertex
        return r2v

    def __generate_extend_ray(self, angle):
        r2v = Ray()
        r2v.origin = self.pos
        r2v.dir_point = self.scaled_rot_vec(self.pos, angle, 10000)
        r2v.fwd = geo.norm_vec((r2v.dir_point[0] - r2v.origin[0], r2v.dir_point[1] - r2v.origin[1]))
        r2v.rot_angle = angle
        r2v.rela_vertex = None
        return r2v

    @staticmethod
    def scaled_rot_vec(start, angle, length):
        rad = angle * DEG2RAD
        return start[0] + math.cos(rad) * length, start[1] + math.sin(rad) * length
