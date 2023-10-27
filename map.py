import numpy as np
from PIL import Image

from primitives import Node

# class RandomMap2D:
#    def __init__(self, size):


class ImageMap2D:
    def __init__(self, image_fn, threshold=0.5):
        self._threshold = threshold
        self._map = self.convert(image_fn)
        self._shape = self._map.shape
        self.parse_free_conf()

    @property
    def map(self):
        return self._map

    @property
    def col(self):
        return self._shape[1]

    @property
    def row(self):
        return self._shape[0]

    @property
    def free_conf(self):
        return self._free_conf

    def convert(self, image_fn):
        image = Image.open(image_fn).convert("L")
        image = np.asarray(image) / 255.0
        map = 1 - (image < self._threshold)
        return map

    def in_range(self, node):
        # black (0) as obstacle and white (1) as free space
        if (
            (node.x < 0)
            or (node.y < 0)
            or node.x >= self.row
            or node.y >= self.col
        ):
            return False
        else:
            return True

    def in_collision(self, x, y):
        # black (0) as obstacle and white (1) as free space
        if self.map[x, y] == 0:
            return True
        else:
            return False

    def line_in_collision(self, start_node, end_node):
        x1 = start_node.x
        y1 = start_node.y
        x2 = end_node.x
        y2 = end_node.y
        in_collision = (
            self.in_collision(x1, y1)
            or self.in_collision(x2, y2)
            or ((x1 == x2) and (y1 == y2))
        )
        if not in_collision:
            n_check = max(abs(x2 - x1), abs(y2 - y1))
            step_size = (float(x2 - x1) / n_check, float(y2 - y1) / n_check)
            for i in range(n_check):
                tmp_x = int(x1 + (i + 1) * step_size[0])
                tmp_y = int(y1 + (i + 1) * step_size[1])
                if self.in_collision(tmp_x, tmp_y):
                    in_collision = True
                    break
        return in_collision

    def parse_free_conf(self):
        self._free_conf = []
        for i in range(self.row):
            for j in range(self.col):
                if not self.in_collision(i, j):
                    self._free_conf.append(Node(i, j, None))

    def is_valid_conf(self, point):
        if (point.x >= self.row) or (point.y >= self.col):
            return False
        elif self.in_collision(point.x, point.y):
            return False
        else:
            return True
