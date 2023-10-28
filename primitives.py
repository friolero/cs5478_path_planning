import numpy as np

from utils import distance


class Node(object):
    def __init__(self, x, y, parent_node):
        self._x = x
        self._y = y
        self.parent = parent_node

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __eq__(self, other):
        return self._x == other.x and self._y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def distance_to(self, other_node):
        return distance(self, other_node)

    def __add__(self, other):
        if isinstance(other, Node):
            return Node(self.x + other.x, self.y + other.y, None)
        else:
            raise TypeError("Unsupported operand type for addition")

    def __sub__(self, other):
        if isinstance(other, Node):
            return Node(self.x - other.x, self.y - other.y, None)
        else:
            raise TypeError("Unsupported operand type for subtraction")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Node(self.x * other, self.y * other, None)
        else:
            raise TypeError("Unsupported operand type for multiplication")

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Node(other * self.x, other * self.y, None)
        else:
            raise TypeError("Unsupported operand type for multiplication")

    def array(self):
        return np.array([self.x, self.y], np.float32)
