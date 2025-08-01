class Node:
    def __init__(self, data, prev_=None, next_=None):
        self.data = data
        self.prev_ = prev_
        self.next_ = next_


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        """
        加到表尾

        Args:
            data:

        Returns:

        """
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev_ = self.tail
            self.tail.next_ = new_node
            self.tail = new_node
        self.size += 1

    def append_before(self, tar_node, data):
        if tar_node is None:
            return
        new_node = Node(data)
        new_node.next_ = tar_node
        if tar_node is self.head:
            self.head = new_node
        else:
            pre_node = tar_node.prev_
            pre_node.next_ = new_node
            new_node.prev_ = pre_node
        tar_node.prev_ = new_node
        self.size += 1

    def prepend(self, data):
        """
        加到表头

        Args:
            data:

        Returns:

        """
        new_node = Node(data, next_=self.head)
        if self.head:
            self.head.prev_ = new_node
        self.head = new_node
        if not self.tail:
            self.tail = new_node
        self.size += 1

    def delete(self, node):
        if node is self.head:
            self.head = node.next_
            if self.head:
                self.head.prev_ = None
        elif node is self.tail:
            self.tail = node.prev_
            if self.tail:
                self.tail.next_ = None
        else:
            node.prev_.next_ = node.next_
            node.next_.prev_ = node.prev_
        self.size -= 1

    def remove(self, data):
        current = self.head
        tar_node = None
        while current is not None:
            if current.data is data:
                tar_node = current
                break
            else:
                current = current.next_

        if tar_node is not None:
            self.delete(tar_node)
            return True
        else:
            return False

    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.data
            current = current.next_
