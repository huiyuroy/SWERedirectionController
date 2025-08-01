from pyrdw.core.space import *


class Trajectory:

    def __init__(self):
        self.id = 0
        self.type = 'abs_road'
        self.tar_num = 0
        self.tar_data = None
        self.__targets = None
        self.__start_idx = 0
        self.__end_idx = -1  # 若大于总长度，自动选择最后一个目标

    def range_targets(self, start=0, end=-1):
        self.__start_idx = start
        self.__end_idx = end
        if self.__end_idx == -1 or self.__end_idx > self.tar_num:
            self.__end_idx = self.tar_num - 1

        self.__targets = self.tar_data[self.__start_idx:self.__end_idx + 1]

    def range_distance(self, max_dis=20000):
        pre_idx = 0
        end_idx = 1
        travel_dis = 0
        while travel_dis < max_dis and end_idx < len(self.tar_data):
            dis = alg.l2_norm(np.array(self.tar_data[end_idx]) - np.array(self.tar_data[pre_idx]))
            travel_dis +=dis
            pre_idx = end_idx
            end_idx += 1
        self.__start_idx = 0
        self.__end_idx = end_idx
        self.__targets = self.tar_data[self.__start_idx:self.__end_idx + 1]

    def walkable(self):
        return np.array(self.__targets)
