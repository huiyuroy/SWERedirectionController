class Node:
    def __init__(self):
        self.pos = [0, 0]
        self.id = -1
        self.child_ids = []
        self.father_ids = []
        self.rela_loop_id = -1
        self.child_nodes = []
        self.father_nodes = []
        self.rela_loop_node = None
        self.visited = False
        self.list_loc = -1

    def print_info(self):
        print(self.pos[0], self.pos[1], self.id, self.child_ids, self.father_ids, self.rela_loop_id)
        child_ids_str = ""
        father_ids_str = ""
        relative_loop_id_str = ""
        for node in self.child_nodes:
            child_ids_str = child_ids_str + str(node.id)
        for node in self.father_nodes:
            father_ids_str = father_ids_str + str(node.id)
        if self.rela_loop_node is not None:
            relative_loop_id_str = self.rela_loop_node.id
        print("child:", child_ids_str, "father:", father_ids_str, "relative_loop:", relative_loop_id_str)
        print("\n")


class Patch:
    def __init__(self):
        self.id = -1
        self.nodes = []
        self.nodes_num = 0
        self.start_node = None
        self.end_node = None
        self.start_nb_patches_id = []
        self.end_nb_patches_id = []
        self.start_nb_patches = []
        self.end_nb_patches = []
        # 判断开始节点所连接其他patch相应的节点顺序 True-其他patch与当前patch共享起点 False-其他patch终点连接当前patch
        self.start_nb_patches_order = []
        # 判断开始节点所连接其他patch相应的节点顺序 True-其他patch与当前patch共享终点 False-其他patch起点连接当前patch
        self.end_nb_patches_order = []
        self.visited = False
        self.user_located = False

    def judge_relation_with_other_patch(self, other_patch_id):
        """
        判断另一个patch与本patch的关系

        :param other_patch_id: 另一个patch的id
        :return:“s”-另一个patch是当前的“start patch”，“e”-另一个patch是当前的“end patch”, “n”-两者不邻接
        """
        for other_patch in self.start_nb_patches:
            if other_patch.id == other_patch_id:
                return "s"
        for other_patch in self.end_nb_patches:
            if other_patch.id == other_patch_id:
                return "e"
        return "n"
