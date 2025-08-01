from pyrdw.core import *


class BaseEnv:
    """
    Base rdw environment. It contains several agents to experience rdw in common virtual and physical spaces. The
    virtual trajectories are necessary in the simulation.
    """

    def __init__(self):
        self.time_step = TIME_STEP
        self.agents = {}
        self.v_scene = None
        self.p_scene = None
        self.v_trajectories = None

    def add_agent(self, agent, name='default'):
        """

        Args:
            agent: a well-constructed agent object.
            name: custom name for the agent, mainly used for log.

        Returns:

        """
        self.agents[name] = agent
        agent.name = name

    def init_agents_state(self, p_loc=None, p_fwd=None, v_loc=None, v_fwd=None):
        for ag in self.agents.values():
            ag.inputer.set_phy_init_state(p_loc, p_fwd, rand=p_loc is None or p_fwd is None)
            ag.inputer.set_vir_init_state(v_loc, v_fwd)

    def set_scenes(self, v_scene, p_scene):
        self.v_scene = v_scene
        self.p_scene = p_scene
        for ag in self.agents.values():
            ag.set_scenes(v_scene, p_scene)

    def set_trajectories(self, trajs):
        self.v_trajectories = trajs

    def prepare(self):
        for ag in self.agents.values():
            ag.prepare()

    def reset(self):
        for ag in self.agents.values():
            ag.reset()

    def step(self):
        """
        Calling all agent update processes.

        Returns:

        """

        all_done = True
        for ag in self.agents.values():
            next_s, done, truncated, info = ag.state_refresh()
            all_done = all_done and done
        return all_done
