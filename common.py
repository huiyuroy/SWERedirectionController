import os

project_path = os.path.abspath(os.path.dirname(__file__))
project_path = project_path[:project_path.find('SWERedirectionController') + len('SWERedirectionController')]
root_path = os.path.dirname(project_path)
rl_model_path = root_path + '\\SWERedirectionController\\rlrdw\\models'
data_path = root_path + '\\polyspaces'
