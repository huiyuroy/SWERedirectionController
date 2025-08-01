import numpy as np
import pygame

from pyrdw.vis.ui import *
import pyrdw.lib.math.geometry as geo

update_freq = 1000
draw_scale = const_env["file_scene_scale"]  # 场景数据中1个单位为现实1cm
human_radius = const_env["human_radius"]
scene_board = const_env["scene_board"]
v_obs_width = 600
vir_obs_sp = [0, 0, v_obs_width, v_obs_width]
p_obs_width = 600
p_obs_sp = [0, 0, p_obs_width, p_obs_width]


class DrawColor(Enum):
    Black = (0, 0, 0)
    White = (255, 255, 255)
    Red = (255, 0, 0)
    Green = (0, 255, 0)
    Blue = (0, 0, 255)
    Yellow = (0, 255, 255)
    DarkGray = (64, 64, 64)
    User = (72, 72, 72)
    LightGray = (225, 225, 225)


key_dict = {
    'a': pygame.K_a, 'b': pygame.K_b, 'c': pygame.K_c, 'd': pygame.K_d, 'e': pygame.K_e, 'f': pygame.K_f,
    'g': pygame.K_g, 'h': pygame.K_h, 'i': pygame.K_i, 'j': pygame.K_j, 'k': pygame.K_k, 'l': pygame.K_l,
    'm': pygame.K_m, 'n': pygame.K_n, 'o': pygame.K_o, 'p': pygame.K_p, 'q': pygame.K_q, 'r': pygame.K_r,
    's': pygame.K_s, 't': pygame.K_t, 'u': pygame.K_u, 'v': pygame.K_v, 'w': pygame.K_w, 'x': pygame.K_x,
    'y': pygame.K_y, 'z': pygame.K_z,
    'A': pygame.K_a, 'B': pygame.K_b, 'C': pygame.K_c, 'D': pygame.K_d, 'E': pygame.K_e, 'F': pygame.K_f,
    'G': pygame.K_g, 'H': pygame.K_h, 'I': pygame.K_i, 'J': pygame.K_j, 'K': pygame.K_k, 'L': pygame.K_l,
    'M': pygame.K_m, 'N': pygame.K_n, 'O': pygame.K_o, 'P': pygame.K_p, 'Q': pygame.K_q, 'R': pygame.K_r,
    'S': pygame.K_s, 'T': pygame.K_t, 'U': pygame.K_u, 'V': pygame.K_v, 'W': pygame.K_w, 'X': pygame.K_x,
    'Y': pygame.K_y, 'Z': pygame.K_z,
}


class RDWWindow:

    def __init__(self, env, name):
        self.p_max_size = (10, 30, 920, 920)
        self.v_max_size = (935, 30, v_obs_width, v_obs_width)
        self.c_max_size = (945, 35 + v_obs_width, v_obs_width, 925 - v_obs_width)
        self.screen_size = (self.v_max_size[0] + self.v_max_size[2] + 5, 960)
        self.env = env
        self.agents = None
        self.back_surf = None
        self.back_clr = (0, 0, 0)

        self.v_shp_ivs = tuple()
        self.p_shp_ivs = tuple()
        self.v_obs_render_sp = None
        self.v_render_pos = None
        self.p_render_pos = None
        self.v_render_center = None
        self.p_render_center = None

        # --------------------------虚实场景绘制相关变量-----------------------------------
        self.v_scene, self.p_scene = None, None
        self.v_surf_size, self.p_surf_size = np.array([500.0, 500.0]), np.array([500.0, 500.0])  # 虚实surface大小
        self.v_surf_size_2, self.p_surf_size_2 = np.array([250.0, 250.0]), np.array([250.0, 250.0])  # 虚实surface大小
        self.v_surf, self.v_surf_bk = None, None  # 绘制区域前景和背景
        self.p_surf, self.p_surf_bk = None, None
        self.v_surf_const = None
        self.p_surf_const = None
        self.v_draw_size, self.v_scale = None, 0
        self.p_draw_size, self.p_scale = None, 0
        self.v_draw_loc, self.p_draw_loc = None, None
        self.v_surf_data, self.p_surf_data = None, None
        # 虚拟空间滑动观察窗口大小，默认800cm * 800cm。主要原因是虚拟surface太大，难以直接全部显示
        self.v_obs_sp = vir_obs_sp
        self.user_r = 0  # 用户标识的半径
        self.ui_font = None
        self.ui_clock = None
        self.lock_fps = 100
        self.fps = 100
        self.render_queue = []
        # --------------------------虚实场景绘制控制变量-----------------------------------
        self.enable_render = True
        self.enable_vir_render, self.enable_phy_render = True, True
        self.enable_ptiling_render = False
        self.enable_pconv_render = False
        self.enable_path_render, self.enable_steer_render = True, True
        self.enable_simu_tar_render, self.enable_walk_traj_render = False, False
        self.enable_reset_render = False
        self.enable_fps_lock = False
        # --------------------------绘制队列-----------------------------------------------
        """
           seq_circles-> 要绘制的位置，记录位置点，每个点记录为(x, y)，即((x1,y1), (x2,y2), .....) n*2
           seq_lines-> 要绘制的向量，记录每对位置点，即((x1s,y1s), (x1e,y1e), (x2s,y2s), (x2e,y2e), ......) 2n*2
           seq_polys-> 要绘制的区域，记录一系列位置点，即(((x1s,y1s), (x1e,y1e), (x2s,y2s), (x2e,y2e), ...), (...), ...)，每个区域
                       用一个包含n个点的列表表示，多个区域最终包含在一个tuple里作为返回值给出。
        """
        self.seq_v_circles = []
        self.seq_v_circles_attri = []  # radius, color
        self.seq_v_lines = []
        self.seq_v_lines_attri = []  # width, color
        self.seq_v_polys = []
        self.seq_v_polys_attri = []  # width, color

        self.seq_p_circles = []
        self.seq_p_circles_attri = []  # radius, color
        self.seq_p_lines = []
        self.seq_p_lines_attri = []  # width, color
        self.seq_p_polys = []
        self.seq_p_polys_attri = []  # width, color
        # -------------------------交互属性-------------------------------------
        self.dragging = False
        self.ms_pos = ()  # 当前鼠标位置
        self.ky_pressed = ()  #
        self.lf_down_pos = ()  # 左键按下位置
        self.lf_up_pos = ()  # 左键按下位置

        # -------------------------ui组件初始化---------------------------------
        self.init_ui(name)

    def init_ui(self, name='rdw play window'):
        pygame.init()
        pygame.display.set_caption(name)
        self.ui_font = pygame.font.SysFont("timesnewroman", 20)
        self.ui_clock = pygame.time.Clock()
        self.back_surf = pygame.display.set_mode(self.screen_size, flags=pygame.DOUBLEBUF, depth=32, display=0, vsync=0)
        self.back_surf.fill((255, 255, 255))
        pygame.draw.rect(self.back_surf, self.back_clr, self.v_max_size, 1)
        pygame.draw.rect(self.back_surf, self.back_clr, self.p_max_size, 1)
        pygame.draw.rect(self.back_surf, self.back_clr, self.c_max_size, 1)

        blits_seq = (
            (self.ui_font.render('Physical Space', True, self.back_clr), (10, 5)),
            (self.ui_font.render('Virtual Space', True, self.back_clr), (935, 5)),
            (self.ui_font.render('Log', True, self.back_clr), (945, 45 + v_obs_width))
        )

        self.back_surf.blits(blits_seq)
        self.ms_pos = pygame.mouse.get_pos()  # 当前鼠标位置
        self.ky_pressed = pygame.key.get_pressed()  #

    def __prepare_surfs(self):
        self.v_scene, self.p_scene = self.env.v_scene, self.env.p_scene
        assert self.v_scene is not None and self.p_scene is not None, 'must set instance to vir and phy scene.'

        v_w, v_h = list(map(lambda x: math.ceil((x + scene_board * 2) * draw_scale), self.v_scene.max_size))
        v_w = v_w if v_w > vir_obs_sp[2] else vir_obs_sp[2] + 10
        v_h = v_h if v_h > vir_obs_sp[3] else vir_obs_sp[3] + 10
        self.v_surf_size = np.array([v_w, v_h], dtype=np.int32)
        self.v_surf_size_2 = self.v_surf_size * 0.5
        self.v_scale = v_obs_width / self.v_surf_size.max()
        self.v_draw_size = self.v_surf_size * self.v_scale
        self.v_draw_loc = (0, 0, *self.v_draw_size)
        self.v_render_center = (self.v_max_size[0] + self.v_max_size[2] * 0.5,
                                self.v_max_size[1] + self.v_max_size[3] * 0.5)
        self.v_surf_const = pygame.Surface(self.v_surf_size)
        self.v_surf_const.fill(DrawColor.Black.value)
        for vb in self.v_scene.bounds:  # 虚拟空间整体surface
            b_ps = (np.array(vb.points) - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2
            color = DrawColor.White.value if vb.is_out_bound else DrawColor.Black.value
            pygame.gfxdraw.aapolygon(self.v_surf_const, b_ps, color)
            pygame.gfxdraw.filled_polygon(self.v_surf_const, b_ps, color)
        self.v_surf_const = pygame.transform.smoothscale(self.v_surf_const, self.v_draw_size)

        p_w, p_h = list(map(lambda x: math.ceil((x + scene_board * 2) * draw_scale), self.p_scene.max_size))
        self.p_surf_size = np.array([p_w, p_h], dtype=np.int32)
        self.p_surf_size_2 = self.p_surf_size * 0.5
        self.p_scale = p_obs_width / self.p_surf_size.max()
        self.p_draw_size = self.p_surf_size * self.p_scale
        self.p_draw_loc = (0, 0, *self.p_draw_size)
        self.p_render_center = (self.p_max_size[0] + self.p_max_size[2] * 0.5,
                                self.p_max_size[1] + self.p_max_size[3] * 0.5)
        self.user_r = int(human_radius * draw_scale * self.v_scale)
        self.p_surf_const = pygame.Surface(self.p_surf_size)
        self.p_surf_const.fill(DrawColor.Black.value)
        for pb in self.p_scene.bounds:  # 物理空间整体surface
            b_ps = (np.array(pb.points) - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2
            color = DrawColor.White.value if pb.is_out_bound else DrawColor.Black.value
            pygame.gfxdraw.aapolygon(self.p_surf_const, b_ps, color)
            pygame.gfxdraw.filled_polygon(self.p_surf_const, b_ps, color)
        if self.enable_pconv_render:
            for conv in self.p_scene.conv_polys:
                c_v = (np.array(conv.vertices) - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2
                pygame.gfxdraw.aapolygon(self.p_surf_const, c_v, DrawColor.DarkGray.value)
                pygame.gfxdraw.filled_polygon(self.p_surf_const, c_v, DrawColor.LightGray.value)
        if self.enable_ptiling_render:
            for t in self.p_scene.tilings:
                c = (255, 255, 255) if t.type else (0, 0, 0)
                t_pos = ((t.center - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2).astype(np.int32)
                pygame.gfxdraw.filled_circle(self.p_surf_const, t_pos[0], t_pos[1], 1, c)
        self.p_surf_const = pygame.transform.smoothscale(self.p_surf_const, self.p_draw_size)

    def __prepare_agents(self):
        ag_colors = ((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                     range(len(self.env.agents.values())))
        self.agents = tuple(zip(self.env.agents.values(), ag_colors))
        for ag, ag_color in self.agents:
            setattr(ag, 'draw_color', ag_color)

    def __render_surfs(self):
        self.v_surf.blit(self.v_surf_bk, (0, 0))
        self.p_surf.blit(self.p_surf_bk, (0, 0))
        for ag, ag_color in self.agents:
            if ag.enable_render:
                ag.render(self, ag.draw_color)
                ag.inputer.render(self, ag.draw_color)
                ag.gainer.render(self, ag.draw_color)
                ag.rdwer.render(self, ag.draw_color)
                ag.resetter.render(self, ag.draw_color)
        self.v_surf_data = pygame.transform.flip(self.v_surf.subsurface(self.v_draw_loc), False, True)
        self.p_surf_data = pygame.transform.flip(self.p_surf.subsurface(self.p_draw_loc), False, True)

    def prepare(self):
        """
        Prepare necessary render elements for env. Must be used before ui.reset().

        Notes: if vir scene or phy scene changes, must recall prepare() function to refresh rendering components.

        Returns:

        """

        self.__prepare_surfs()
        self.__prepare_agents()

    def reset(self):
        self.lock_fps = 0 if not self.enable_fps_lock else 100
        self.v_surf = self.v_surf_const.copy()
        self.v_surf_bk = self.v_surf_const.copy()
        self.p_surf = self.p_surf_const.copy()
        self.p_surf_bk = self.p_surf_const.copy()
        self.v_render_pos = (self.v_render_center[0] - self.v_draw_loc[2] * 0.5,
                             self.v_render_center[1] - self.v_draw_loc[3] * 0.5)
        self.p_render_pos = (self.p_render_center[0] - self.p_draw_loc[2] * 0.5,
                             self.p_render_center[1] - self.p_draw_loc[3] * 0.5)
        self.__render_surfs()

    def render(self):
        """
        draw virtual and physical spaces on the surface, and update other elements of the surface.

        Returns:

        """
        # -------monitor surface events----------------------------------------------------------
        self.sys_control_listen()
        self.fps = self.ui_clock.get_fps()
        self.render_queue = []
        if self.enable_render:
            # ---------------------------------------------------ui update---------------------------------------------
            self.__render_surfs()
            # --------------------------------------------------frame blits--------------------------------------------
            self.render_queue.append((self.v_surf_data, self.v_render_pos))  # self.v_render_pos
            self.render_queue.append((self.p_surf_data, self.p_render_pos))

        self.back_surf.fill((255, 255, 255), self.c_max_size)
        self.render_queue.append(
            (self.ui_font.render('fps:{:.2f}'.format(self.fps), True, self.back_clr), (945, 55 + v_obs_width)))

        ag_idx = 0
        ui_ox, ui_oy = 945, 55 + v_obs_width
        for ag, ag_color in self.agents:
            if ag.enable_render:
                ag_x = ui_ox + ag_idx * 150
                render_data = [
                    (f'name:{ag.name}', 30),
                    (f'resets:{ag.resetter.reset_num}', 60),
                    # (f'rdw rate:{ag.rdw_rate:.2f}', 90),
                ]
                for text, offset_y in render_data:
                    self.render_queue.append((self.ui_font.render(text, True, self.back_clr), (ag_x, ui_oy + offset_y)))
                ag_idx += 1

        self.back_surf.blits(self.render_queue)
        pygame.display.flip()
        self.ui_clock.tick(self.lock_fps)

    def render_mode(self, enable=True):
        self.enable_render = enable
        for ag, _ in self.agents:
            ag.enable_render = enable

    def render_custom(self):
        pass

    # -----------------------------------------render methods-----------------------------------------------------------
    #
    # To render other contents on the screen, use these call back methods.
    # Note: all color are in RGBA mode, e.g.,
    # red->(255, 0, 0, 255).
    # To set a transport color, set the last value a between [0, 255].
    #
    # ------------------------------------------------------------------------------------------------------------------

    def draw_vir_circle(self, c, r, color=None):
        pos = (((c - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(np.int32)
        pygame.gfxdraw.aacircle(self.v_surf, pos[0], pos[1], r, color if color is not None else (0, 0, 0))

    def draw_vir_circle_bg(self, c, r, color=None):
        pos = (((c - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(np.int32)
        pygame.gfxdraw.aacircle(self.v_surf_bk, pos[0], pos[1], r, color if color is not None else (0, 0, 0))

    def draw_phy_circle(self, c, r, color=None):
        pos = (((c - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(np.int32)
        pygame.gfxdraw.aacircle(self.p_surf, pos[0], pos[1], int(r * draw_scale * self.p_scale),
                                color if color is not None else (0, 0, 0))

    def draw_phy_circle_bg(self, c, r, color=None):
        pos = (((c - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(np.int32)
        pygame.gfxdraw.aacircle(self.p_surf, pos[0], pos[1], int(r * draw_scale * self.p_scale),
                                color if color is not None else (0, 0, 0))

    def draw_vir_line(self, s, e, w, color=None):
        s_pos = (((s - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(np.int32)
        e_pos = (((e - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(np.int32)
        pygame.draw.line(self.v_surf, color if color is not None else (0, 0, 0), s_pos, e_pos, w)

    def draw_vir_line_bg(self, s, e, w, color=None):
        s_pos = (((s - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(np.int32)
        e_pos = (((e - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(np.int32)
        pygame.draw.line(self.v_surf_bk, color if color is not None else (0, 0, 0), s_pos, e_pos, w)

    def draw_phy_line(self, s, e, w, color=None):
        s_pos = (((s - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(np.int32)
        e_pos = (((e - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(np.int32)
        pygame.draw.line(self.p_surf, color if color is not None else (0, 0, 0), s_pos, e_pos, w)

    def draw_phy_line_bg(self, s, e, w, color=None):
        s_pos = (((s - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(np.int32)
        e_pos = (((e - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(np.int32)
        pygame.draw.line(self.p_surf_bk, color if color is not None else (0, 0, 0), s_pos, e_pos, w)

    def draw_vir_poly(self, vertexes, fill, color=None):
        """

        Args:
            vertexes: 顶点序列，e.g., ((x1,y1),(x2,y2),...)
            fill:
            color:

        Returns:

        """
        vertexes = (((np.array(
            vertexes) - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(np.int32)
        pygame.gfxdraw.aapolygon(self.v_surf, vertexes, color)
        if fill:
            pygame.gfxdraw.filled_polygon(self.v_surf, vertexes, color)

    def draw_phy_poly(self, vertexes, fill, color=None):
        """

        Args:
            vertexes: 顶点序列，e.g., ((x1,y1),(x2,y2),...)
            fill:
            color: (0~255,0~255,0~255, alpha)

        Returns:

        """

        vertexes = (((np.array(
            vertexes) - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(np.int32)
        pygame.gfxdraw.aapolygon(self.p_surf, vertexes, color)
        if fill:
            pygame.gfxdraw.filled_polygon(self.p_surf, vertexes, color)

    # ---------------------------------------mouse keyboard-------------------------------------------------------------
    def sys_control_listen(self):
        self.ms_pos = pygame.mouse.get_pos()
        self.ky_pressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 鼠标按下，开始拖拽
                self.dragging = True
                self.lf_down_pos = self.ms_pos
            elif event.type == pygame.MOUSEBUTTONUP:
                # 鼠标释放，停止拖拽
                self.dragging = False
                self.lf_up_pos = self.ms_pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.enable_render = not self.enable_render

    def trans_pos_from_psurf(self, pos) -> Tuple:
        x_off = (pos[0] - self.p_render_center[0]) / self.p_scale
        y_off = (-pos[1] + self.p_render_center[1]) / self.p_scale
        px = (self.p_surf_size_2[0] + x_off) / draw_scale - scene_board
        py = (self.p_surf_size_2[1] + y_off) / draw_scale - scene_board
        return px, py

    def get_mouse_pos(self):
        return self.ms_pos

    def get_key_pressed(self, key='k'):
        return self.ky_pressed[key_dict[key]]

    def obtain_mk_live_input(self):
        p_scene_tar_pos = self.trans_pos_from_psurf(self.ms_pos) if self.dragging else None
        w_down = self.ky_pressed[pygame.K_w]
        s_down = self.ky_pressed[pygame.K_s]
        return p_scene_tar_pos, w_down, s_down, self.fps
