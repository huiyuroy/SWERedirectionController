from pyrdw import default
from pyrdw import generator
from pyrdw.core.agent.base import GeneralRdwAgent
from rlrdw.swerc.swerc_benchmark import SWERCRDWAgent, SWERCDiscreteScene


def load_scene(tar_path,
               simple_load=False,
               load_road=False,
               load_vis=True,
               load_extend=True,
               scene_class='base'):
    if 'base' in scene_class:
        return generator.load_scene(tar_path, simple_load, load_road, load_vis, load_extend)
    elif 'swerc' in scene_class:
        return generator.load_scene(tar_path, simple_load, load_road, load_vis, load_extend, SWERCDiscreteScene)
    return None


def obtain_agent(gainer='simple', rdwer='no', resetter='r21', inputer='traj', agent_manager='general', **kwargs):
    # -----------------设定增益管理器--------------------------------
    tar_gain = default.__obtain_gain_manager(gainer)
    tar_gain.load_params()
    # -----------------设定重定向管理器--------------------------------
    tar_rdw =default.__obtain_rdw_manager(rdwer)
    tar_rdw.load_params()
    # -----------------设定重置管理器--------------------------------
    tar_reset = default.__obtain_reset_manager(resetter)
    tar_reset.load_params()
    # -----------------设定行走管理器--------------------------------
    tar_inputer = default.__obtain_input_manager(inputer)
    tar_inputer.load_params()

    tar_agent = None
    if 'general' in agent_manager:
        tar_agent = GeneralRdwAgent()
    elif 'swerc' in agent_manager:
        tar_agent = SWERCRDWAgent(input_type=inputer,
                                  reset_type=kwargs['swerc_reset_type'])

    tar_agent.set_manager(tar_gain, 'gain')
    tar_agent.set_manager(tar_rdw, 'rdw')
    tar_agent.set_manager(tar_reset, 'reset')
    tar_agent.set_manager(tar_inputer, 'inputer')

    return tar_agent