from pyrdw.core.env import *
from pyrdw.util.log import RDWTrajectoryWalkerLogger
from pyrdw.vis.ui.base import RDWWindow


class RdwEnv(BaseEnv):
    """
    The rdw env containing ui and log. Recommend to modify when designing specific environment.

    """

    def __init__(self):
        super().__init__()
        self.env_ui: RDWWindow = None
        self.env_logs = {}

    def add_logger(self, agent_name, logger):
        logger.set_agent(self.agents[agent_name])
        logger.set_epi_data_buffer(100)
        self.env_logs[agent_name] = logger

    def load_logger(self):
        for ag in self.agents.values():
            ag_log = RDWTrajectoryWalkerLogger(ag)
            ag_log.set_agent(ag)
            ag_log.set_epi_data_buffer(100)
            self.env_logs[ag.name] = ag_log

    def load_ui(self, name='rdw play window', UIClass=None):
        if UIClass is None:
            self.env_ui = RDWWindow(self, name)
        else:
            self.env_ui = UIClass(self, name)

    def set_current_trajectory(self, traj):
        """
        将所有当前的agent模拟路线设置为指定的路径

        Args:
            traj: 指定路径对象

        Returns:

        """
        for ag in self.agents.values():
            ag.inputer.set_trajectory(traj)

    def prepare(self):
        super().prepare()
        self.env_ui.prepare()
        for ag_log in self.env_logs.values():
            ag_log.prepare_log()

    def reset(self):
        super().reset()
        self.env_ui.reset()
        for ag_log in self.env_logs.values():
            ag_log.reset_epi_data()

    def step(self):
        all_done = super().step()
        return all_done

    def render(self):
        """
        call this function to render the env.

        Returns:

        """
        self.env_ui.render()

    def receive(self):
        """
        Receive input from out device, e.g., mouse keyboard or network. Default is mouse keyboard.

        Note: - only accessible when setting one agent to env
              - the inputer type must be live.

        Returns:

        """
        agent = list(self.agents.values())[0]

        mouse_phy_loc, w_key_down, s_key_down, fps = self.env_ui.obtain_mk_live_input()

        delta_time = 1 / fps if fps != 0 else 1 / 500

        agent.inputer.set_time_step(delta_time)
        vel = 0
        if w_key_down:
            vel = 200
        elif s_key_down:
            vel = -200

        if mouse_phy_loc is not None:
            pfwd = np.array(mouse_phy_loc) - agent.p_cur_loc
        else:
            pfwd = agent.p_cur_fwd
        ploc = geo.norm_vec(pfwd) * vel * delta_time + agent.p_cur_loc

        agent.inputer.set_phy_state(ploc, pfwd)

    def record(self):
        """
        Record log data of each step. Called after step().

        Returns:

        """

        for ag_log in self.env_logs.values():
            ag_log.record_step_data()

    def output_epi_info(self):
        return [ag_log.log_epi_data() for ag_log in self.env_logs.values()]
