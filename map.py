import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import Node

# class RandomMap2D:
#    def __init__(self, size):


class ImageMap2D:
    def __init__(self, image_fn, threshold=0.5, distance_field=False):
        self._threshold = threshold
        self._map = self.convert(image_fn)
        self._shape = self._map.shape
        self.parse_free_conf()
        if distance_field:
            self.parse_distance_field()
        else:
            self._distance_field = None

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

    @property
    def obstacle_conf(self):
        return self._obstacle_conf

    @property
    def distance_field(self):
        return self._distance_field

    @property
    def distance_field_vec(self):
        return self._distance_field_vec

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
        self._obstacle_conf = []
        for i in range(self.row):
            for j in range(self.col):
                if not self.in_collision(i, j):
                    self._free_conf.append(Node(i, j, None))
                else:
                    self._obstacle_conf.append(Node(i, j, None))

    def is_valid_conf(self, point):
        if (point.x >= self.row) or (point.y >= self.col):
            return False
        elif self.in_collision(point.x, point.y):
            return False
        else:
            return True

    def nearest_freespace(self, x, y):
        assert self.in_collision(
            x, y
        ), "Only check nearest freespace for obstacle node"
        radius = 0
        found_freespace = False
        while not found_freespace:
            radius += 1
            sweep_offset = list(range(-radius, radius + 1))
            min_dist = np.sqrt(self.row ** 2 + self.col ** 2 + 1)
            field_vec = None
            for x_offset in [-radius, radius]:
                for y_offset in sweep_offset:
                    tmp_node = Node(x + x_offset, y + y_offset, None)
                    if not self.in_range(tmp_node):
                        continue
                    if not self.in_collision(x + x_offset, y + y_offset):
                        found_freespace = True
                        dist = np.sqrt(x_offset ** 2 + y_offset ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            field_vec = [x_offset, y_offset]
            for y_offset in [-radius, radius]:
                for x_offset in sweep_offset:
                    tmp_node = Node(x + x_offset, y + y_offset, None)
                    if not self.in_range(tmp_node):
                        continue
                    if not self.in_collision(x + x_offset, y + y_offset):
                        found_freespace = True
                        dist = np.sqrt(x_offset ** 2 + y_offset ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            field_vec = [x_offset, y_offset]
        return min_dist, np.array(field_vec, dtype=np.float32)

    def nearest_obstacle(self, x, y):
        if self.in_collision(x, y):
            min_dist, field_vec = self.nearest_freespace(x, y)
            return -min_dist, -field_vec
        radius = 0
        found_obstacle = False
        while not found_obstacle:
            radius += 1
            sweep_offset = list(range(-radius, radius + 1))
            min_dist = np.sqrt(self.row ** 2 + self.col ** 2 + 1)
            field_vec = None
            for x_offset in [-radius, radius]:
                for y_offset in sweep_offset:
                    tmp_node = Node(x + x_offset, y + y_offset, None)
                    if not self.in_range(tmp_node):
                        continue
                    if self.in_collision(x + x_offset, y + y_offset):
                        found_obstacle = True
                        dist = np.sqrt(x_offset ** 2 + y_offset ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            field_vec = [x_offset, y_offset]
            for y_offset in [-radius, radius]:
                for x_offset in sweep_offset:
                    tmp_node = Node(x + x_offset, y + y_offset, None)
                    if not self.in_range(tmp_node):
                        continue
                    if self.in_collision(x + x_offset, y + y_offset):
                        found_obstacle = True
                        dist = np.sqrt(x_offset ** 2 + y_offset ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            field_vec = [x_offset, y_offset]
        return min_dist, np.array(field_vec, dtype=np.float32)

    def parse_distance_field(self):
        print("==> Parsing distance field")
        self._distance_field = np.zeros(self._map.shape)
        self._distance_field_vec = np.zeros(self._map.shape + (2,))
        for x in range(self.row):
            for y in range(self.col):
                self._distance_field[x, y] = self.nearest_obstacle(x, y)[0]
                self._distance_field_vec[x, y] = self.nearest_obstacle(x, y)[1]

    def build_map_cost_grad(self, tol_radius, vis=False):
        print("==> Building map costs and cost gradients...")
        self._col_cost = np.zeros((self.row, self.col), np.float32)
        self._col_cost_grad = np.zeros((self.row, self.col, 2), np.float32)

        for i in range(self.row):
            for j in range(self.col):
                dist, curr2obs_vec = self.nearest_obstacle(i, j)
                col_cost = max(tol_radius - dist, 0)
                self._col_cost[i, j] = col_cost
        self._col_cost_grad = self.finite_difference(self._col_cost)
        print("==> Done!")
        if vis:
            plt.imshow(self._col_cost)
            r, c = self._col_cost_grad.shape[:2]
            Y, X = np.mgrid[0:r, 0:c]
            dy = self._col_cost_grad[..., 0]
            dx = -self._col_cost_grad[..., 1]

            n = 2
            plt.quiver(X[::n, ::n], Y[::n, ::n], dx[::n, ::n], dy[::n, ::n])
            plt.show()

    def finite_difference(self, value):
        grad = np.zeros(value.shape + (len(value.shape),))

        for i in range(value.shape[0]):
            max_j = value.shape[1] - 1
            if i == 0:
                grad[i, 0] = [
                    value[i + 1, 0] - value[i, 0],
                    value[i, 1] - value[i, 0],
                ]
                grad[i, max_j] = [
                    value[i + 1, max_j] - value[i, max_j],
                    value[i, max_j] - value[i, max_j - 1],
                ]
            elif i == (value.shape[0] - 1):
                grad[i, 0] = [
                    value[i, 0] - value[i - 1, 0],
                    value[i, 1] - value[i, 0],
                ]
                grad[i, max_j] = [
                    value[i, max_j] - value[i - 1, max_j],
                    value[i, max_j] - value[i, max_j - 1],
                ]
            else:
                grad[i, 0] = [
                    (value[i + 1, 0] - value[i - 1, 0]) / 2,
                    value[i, 1] - value[i, 0],
                ]
                grad[i, max_j] = [
                    (value[i + 1, max_j] - value[i - 1, max_j]) / 2,
                    value[i, max_j] - value[i, max_j - 1],
                ]

        for j in range(value.shape[1]):
            max_i = value.shape[0] - 1
            if j == 0:
                grad[0, j] = [
                    value[1, j] - value[0, j],
                    value[0, j + 1] - value[0, j],
                ]
                grad[max_i, j] = [
                    value[max_i, j] - value[max_i - 1, j],
                    value[max_i, j + 1] - value[max_i, j],
                ]
            elif j == (value.shape[1] - 1):
                grad[0, j] = [
                    value[1, j] - value[0, j],
                    value[0, j] - value[0, j - 1],
                ]
                grad[max_i, j] = [
                    value[max_i, j] - value[max_i - 1, j],
                    value[max_i, j] - value[max_i, j - 1],
                ]
            else:
                grad[0, j] = [
                    value[1, j] - value[0, j],
                    (value[0, j + 1] - value[0, j - 1]) / 2,
                ]
                grad[max_i, j] = [
                    value[max_i, j] - value[max_i - 1, j],
                    (value[max_i, j + 1] - value[max_i, j - 1]) / 2,
                ]

        for i in range(1, value.shape[0] - 2):
            for j in range(1, value.shape[1] - 2):
                grad[i, j] = [
                    (value[i + 1, j] - value[i - 1, j]) / 2,
                    (value[i, j + 1] - value[i, j - 1]) / 2,
                ]
        return grad

    @property
    def col_cost(self):
        return self._col_cost

    @property
    def col_cost_grad(self):
        return self._col_cost_grad
