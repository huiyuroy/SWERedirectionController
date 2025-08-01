import random

import numpy as np

from pyrdw.core.gain import *


class SimpleGainManager(BaseGainManager):

    def __init__(self):
        super().__init__()

    def reset(self):
        self.g_values = self.g_pri[0]

    def update(self, **kwargs):
        self.g_values += self.g_rates
        return self.g_values

    def render(self, wdn_obj, default_color):
        pass


class LinearGainManager(BaseGainManager):

    def __init__(self):
        super().__init__()

    def reset(self):
        self.g_values = self.g_pri[0]

    def update(self, **kwargs):
        self.g_values += self.g_rates
        return self.g_values

    def render(self, wdn_obj, default_color):
        pass




