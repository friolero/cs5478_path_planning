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

    def in_collision(self, x, y):
        # black (0) as obstacle and white (1) as free space
        if self.map[x, y] == 0:
            return True
        else:
            return False

    def parse_free_conf(self):
        self._free_conf = []
        for i in range(self.row):
            for j in range(self.col):
                if not self.in_collision(i, j):
                    self._free_conf.append(Node(i, j))

    def is_valid_conf(self, point):
        if (point.x >= self.row) or (point.y >= self.col):
            return False
        elif self.in_collision(point.x, point.y):
            return False
        else:
            return True
