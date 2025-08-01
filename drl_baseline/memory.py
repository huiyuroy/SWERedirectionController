import random
from abc import ABC, abstractmethod
import numpy as np
import torch

data_type = torch.float32

DEFALT_DATA_INFO = {
    's': [(1,), torch.float32],
    'im_s': [(1, 1, 1), torch.float32],
    'a': [(1,), torch.float32],
    'r': [(1,), torch.float32],
    's_': [(1,), torch.float32],
    'im_s_': [(1, 1, 1), torch.float32],
    'd': [(1,), torch.bool],
    'dw': [(1,), torch.bool]

}


class Memory(ABC):
    """
    Abstract Memory class to define the API methods
    """

    def __init__(self,
                 max_size: int,
                 batch_size: int,
                 memory_info: dict,
                 device):
        """
        use data_info to generate builtin data buffer. Each data item corresponds to a distinct memory.

        Args:
            max_size:
            batch_size:
            memory_info: should be in the following format
                {'x': [(x_dim, ..), d_type] ....... }
                size: (dim, ) 'vector' or (C x H x W) 'image'
            device:
        """

        self.device = device
        self.max_size = max_size
        self.mem_size = 0
        self.mem_ptr = -1
        self.batch_size = batch_size
        self.batch_idx = None
        self.batch = None
        self.batch_ready = False
        # declare what data keys to store
        self.memory_info = memory_info
        self.data = dict()

    @abstractmethod
    def reset_memory(self):
        """
        Fully reset the memory storage and related variables

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def store_data(self, **kwargs):
        """
        Implement memory update given the full info from the latest timestep. NOTE: guard for np.nan reward and
        done when individual env resets.

        Args:
            **kwargs: para name should be exactly same as the keys of data_info

        Returns:

        """

        raise NotImplementedError

    @abstractmethod
    def ready(self):
        """
        Check if the memory is ready for sample

        Returns:
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        """
        Implement memory sampling mechanism

        Returns:
        """

        raise NotImplementedError


class ReplayMemory(Memory):
    """
    The definition of replay_memory, need to specify the max_size and batch_size.

    For example:
        rm = ReplayMemory(
                            max_size=int(1e6),
                            batch_size=self.batch_size,
                            data_info={ 's': [(state_dim,), torch.float],
                                        'a': [(action_dim,), torch.float],
                                        'r': [(1,), torch.float],
                                        's_': [(state_dim,), torch.float],
                                        'dw': [(1,), torch.bool]},
                            device=self.dvc
        )
    """

    def __init__(self,
                 max_size: int,
                 batch_size: int,
                 memory_info,
                 use_latest: bool,
                 device):

        super().__init__(max_size, batch_size, memory_info, device)
        if memory_info is None:
            self.memory_info = DEFALT_DATA_INFO
        self.use_latest = use_latest  # whether you use the latest data for batch

    def reset_memory(self):
        self.mem_ptr = -1
        self.mem_size = 0
        self.data = dict()
        for data_name, data_detail in self.memory_info.items():
            data_dim, d_type = data_detail
            data_buffer = np.zeros((self.max_size, *data_dim), dtype=d_type)
            self.data[data_name] = data_buffer
        self.batch = {}
        self.batch_ready = False

    def store_data(self, **kwargs):
        """

        Args:
            **kwargs:

        Returns:

        """
        self.mem_ptr = (self.mem_ptr + 1) % self.max_size
        for k, v in kwargs.items():
            self.data[k][self.mem_ptr] = v
        if self.mem_size < self.max_size:
            self.mem_size += 1
        self.batch_ready = self.mem_size >= self.batch_size

    def ready(self):
        return self.batch_ready

    def sample(self):
        if self.batch_ready:
            self.batch_idx = self.__sample_idx()
            for k in self.data.keys():
                self.batch[k] = self.data[k][self.batch_idx]
            return self.batch
        else:
            return None

    def __sample_idx(self):
        # batch_idx = np.random.choice(self.mem_size, self.batch_size, replace=False).astype(int)
        batch_idx = self.random_unique_integers(0, self.mem_size - 1, self.batch_size)
        if self.use_latest:  # add the latest sample
            batch_idx[-1] = self.mem_ptr
        return batch_idx

    @staticmethod
    def random_unique_integers(start, end, count):
        nums = set()
        while len(nums) < count:
            nums.add(random.randint(start, end))
        return np.array(list(nums))


class PrioritizedReplayMemory(ReplayMemory):
    """
    Prioritized Experience Replay

    Implementation follows the approach in the paper "Prioritized Experience Replay", Schaul et al 2015" https://arxiv.org/pdf/1511.05952.pdf and is Jaromír Janisch's with minor adaptations.
    See memory_util.py for the license and link to Jaromír's excellent blog

    Stores agent experiences and samples from them for agent training according to each experience's priority

    The memory has the same behaviour and storage structure as Replay memory with the addition of a SumTree to store and sample the priorities.

    e.g. memory_spec
    "memory": {
        "name": "PrioritizedReplay",
        "alpha": 1,
        "epsilon": 0,
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    """

    def __init__(self,
                 max_size: int = int(1e6),
                 batch_size: int = 256,
                 memory_info=None,
                 use_latest: bool = False,
                 device='cpu'):
        super().__init__(max_size, batch_size, memory_info, use_latest, device)
        self.priorities = None
        self.tree_idx = None
        self.tree = None
        self.epsilon = np.full((1,), 0.0001)
        self.alpha = np.full((1,), 0.6)
        self.error = 100000

    def reset_memory(self):
        super().reset_memory()
        self.priorities = [None] * self.max_size
        self.tree = SumTree(self.max_size)

    def store_data(self, **kwargs):
        """
        Implementation for update() to add experience to memory, expanding the memory size if necessary.
        All experiences are added with a high priority to increase the likelihood that they are sampled at least once.
        """
        super().store_data(**kwargs)
        priority = self.get_priority(self.error)
        self.priorities[self.mem_ptr] = priority
        self.tree.add(priority, self.mem_ptr)

    def get_priority(self, error):
        """
        Takes in the error of one or more examples and returns the proportional priority
        """
        return np.power(error + self.epsilon, self.alpha).squeeze()

    def sample_idx(self):
        """
        Samples batch_size indices from memory in proportional to their priority.
        """
        batch_idx = np.zeros(self.batch_size)
        self.tree_idx = np.zeros(self.batch_size, dtype=np.int32)

        for i in range(self.batch_size):
            s = random.uniform(0, self.tree.total())
            tree_i, p, idx = self.tree.get(s)
            batch_idx[i] = idx
            self.tree_idx[i] = tree_i

        batch_idx = batch_idx.astype(int)
        if self.use_latest:  # add the latest sample
            batch_idx[-1] = self.mem_ptr
        return batch_idx

    def update_priorities(self, errors):
        """
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        """
        priorities = self.get_priority(errors)
        assert len(priorities) == self.batch_idx.size
        for idx, p in zip(self.batch_idx, priorities):
            self.priorities[idx] = p
        for p, i in zip(priorities, self.tree_idx):
            self.tree.update(i, p)


class SumTree:
    """
        This SumTree code is a modified version and the original code is from:
        https://github.com/jaara/AI-blog/blob/master/SumTree.py
        Story data with its priority in the tree.
    """

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.write = 0
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.indices = np.zeros(capacity)  # Stores the indices of the experiences
        # [--------------indices frame-------------]
        #             size: capacity

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1
        self.indices[self.write] = index
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        index_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.indices[index_idx]

    def print_tree(self):
        for i in range(len(self.indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.indices[i]}, Prio: {self.tree[j]}')
