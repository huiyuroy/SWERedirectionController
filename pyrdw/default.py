import pyrdw.core.agent.base as agent_base
import pyrdw.core.gain.base as gain_base
import pyrdw.core.rdwer.reactive as rdw_react
import pyrdw.core.resetter.base as reset_base
import pyrdw.core.input.base as walker_base

from pyrdw.core.space.scene import Scene, DiscreteScene
from pyrdw.core.space.boundary import Boundary
from pyrdw.core.space.roadmap import Node, Patch
from pyrdw.core.space.grid import Tiling
from pyrdw.core.space.trajectory import Trajectory


def obtain_agent(gainer='simple', rdwer='no', resetter='r21', inputer='traj', agent_manager='general'):
    """
    生成指定的重定向agent，可自行组合不同组件。注：部分特定类型agent必须设置特定组件。支持的重定向agent包括：

    Args:
        gainer:
            - simple
            - linear

        rdwer:
            - no
            - s2c
            - s2o
            - apf

        resetter:
            - r21
            - r2g
            - sfr2g
            - rarc
            - tr2c
            - mr2c
        inputer:
            - traj
            - live
        agent_manager:
            - general: 常规agent，无需特定设置，可与不需要特殊离线处理的增益控制器、重定向控制器和重置控制器组合；

    Returns:

    """
    # -----------------设定增益管理器--------------------------------
    tar_gain = __obtain_gain_manager(gainer)
    tar_gain.load_params()

    # -----------------设定重定向管理器--------------------------------
    tar_rdw = __obtain_rdw_manager(rdwer)
    tar_rdw.load_params()

    # -----------------设定重置管理器--------------------------------
    tar_reset = __obtain_reset_manager(resetter)
    tar_reset.load_params()

    # -----------------设定行走管理器--------------------------------
    tar_inputer = __obtain_input_manager(inputer)
    tar_inputer.load_params()

    tar_agent = agent_base.GeneralRdwAgent()
    tar_agent.set_manager(tar_gain, 'gain')
    tar_agent.set_manager(tar_rdw, 'rdw')
    tar_agent.set_manager(tar_reset, 'reset')
    tar_agent.set_manager(tar_inputer, 'inputer')

    return tar_agent


def __obtain_gain_manager(gainer='simple'):
    """

    Args:
        gainer:
         - simple
         - linear
         - arc
         - apfs2t

    Returns:
        target gain manager
    """
    tar_gain = None
    if 'simple' in gainer:
        tar_gain = gain_base.SimpleGainManager()
    elif 'linear' in gainer:
        tar_gain = gain_base.LinearGainManager()

    return tar_gain


def __obtain_rdw_manager(rdwer='no'):
    """

    Args:
        rdwer:
            - no
            - s2c
            - s2o
            - apf
            - arc
            - apfs2t

    Returns:

    """
    tar_rdw = None
    if rdwer == 'no':
        tar_rdw = rdw_react.NoRdwManager()
    elif rdwer == 's2c':
        tar_rdw = rdw_react.S2CRdwManager()
    elif rdwer == 's2o':
        tar_rdw = rdw_react.S2ORdwManager()
    elif rdwer == 'apf':
        tar_rdw = rdw_react.APFRdwManager()

    return tar_rdw


def __obtain_reset_manager(resetter='r21'):
    """

    Args:
        resetter:
            - r21
            - r2g
            - sfr2g
            - rarc
            - tr2c
            - mr2c
            - r2t
            - r2wg
            - r2rc
            - r2mrc
            - r2mpe
            - apfs2t
    Returns:

    """
    tar_reset = None
    if 'r21' == resetter:
        tar_reset = reset_base.Turn21Resetter()
    elif 'r2g' == resetter:
        tar_reset = reset_base.TurnAPFGradientResetter()
    elif 'sfr2g' == resetter:
        tar_reset = reset_base.TurnAPFGradientStepForwardResetter()
    elif 'tr2c' == resetter:  # traditional r2c
        tar_reset = reset_base.TurnCenterResetter()
    elif 'mr2c' == resetter:
        tar_reset = reset_base.TurnModifiedCenterResetter()
    elif 'r2t' == resetter:
        tar_reset = reset_base.TurnSteerForceResetter()


    return tar_reset


def __obtain_input_manager(inputer='traj'):
    """

    Args:
        inputer:
            - traj
            - live
    Returns:

    """
    tar_inputer = None
    if 'traj' in inputer:
        tar_inputer = walker_base.SimuTrajectoryInputer()
    elif 'live' in inputer:
        tar_inputer = walker_base.LiveInputer()

    return tar_inputer
