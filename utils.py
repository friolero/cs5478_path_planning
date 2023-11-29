import collections
import random
from copy import deepcopy

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def vis_map(map):
    image = Image.fromarray((map.map * 255.0).astype(np.uint8))
    image.show()


def vis_path(map, path, r_p=1):

    vis_image = deepcopy(map.map)
    vis_image = vis_image[..., np.newaxis].repeat(3, axis=-1)

    start_node = path[0]
    end_node = path[-1]
    vis_image[
        max(0, start_node.x - r_p) : min(start_node.x + r_p, map.row - 1),
        max(0, start_node.y - r_p) : min(start_node.y + r_p, map.col - 1),
    ] = [1, 0, 0]

    for i, node in enumerate(path[1:]):
        vis_image[
            max(0, node.x - r_p) : min(node.x + r_p, map.row - 1),
            max(0, node.y - r_p) : min(node.y + r_p, map.col - 1),
        ] = [0.0, 1.0, 0.0]

    vis_image[
        max(0, end_node.x - r_p) : min(end_node.x + r_p, map.row - 1),
        max(0, end_node.y - r_p) : min(end_node.y + r_p, map.col - 1),
    ] = [0, 0, 1]

    vis_image = Image.fromarray((vis_image * 255.0).astype(np.uint8))
    vis_image.show()


def save_vis_paths(map, paths, out_fn, r_p=1):

    vis_image = deepcopy(map.map).astype(np.float32)
    vis_image = vis_image[..., np.newaxis].repeat(3, axis=-1)

    jet = plt.get_cmap("jet")
    color_norm = colors.Normalize(vmin=0, vmax=len(paths[0]))
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=jet)

    for path in paths:
        start_node = path[0]
        end_node = path[-1]
        vis_image[
            max(0, start_node[0] - r_p) : min(start_node[0] + r_p, map.row - 1),
            max(0, start_node[1] - r_p) : min(start_node[1] + r_p, map.col - 1),
        ] = [0.0, 0.0, 1.0]

        # rnd_color = np.random.uniform(0.0, 1.0, 3).tolist()
        for i, node in enumerate(path[1:]):
            vis_image[
                max(0, node[0] - r_p) : min(node[0] + r_p, map.row - 1),
                max(0, node[1] - r_p) : min(node[1] + r_p, map.col - 1),
            ] = scalar_map.to_rgba(i)[:3]

        vis_image[
            max(0, end_node[0] - r_p) : min(end_node[0] + r_p, map.row - 1),
            max(0, end_node[1] - r_p) : min(end_node[1] + r_p, map.col - 1),
        ] = [1.0, 0.0, 0.0]

    vis_image = Image.fromarray((vis_image * 255.0).astype(np.uint8))
    if out_fn is not None:
        vis_image.save(out_fn)
    else:
        vis_image.show()


def distance(point_a, point_b):
    return np.linalg.norm([point_a.x - point_b.x, point_a.y - point_b.y])


def sort_with_distance(node, neighbor_nodes):
    dists = [distance(node, neighbor_node) for neighbor_node in neighbor_nodes]
    dists = np.array(dists).reshape(-1)
    nearest_indices = np.argsort(dists, axis=-1).tolist()
    nearest_dists = [dists[idx] for idx in nearest_indices]
    return nearest_indices[0], nearest_dists[0]


def knn(node, neighbor_nodes, k):
    assert len(neighbor_nodes) > k, "Not enough neighbors."
    dists = [distance(node, neighbor_node) for neighbor_node in neighbor_nodes]
    dists = np.array(dists).reshape(-1)
    nearest_indices = np.argsort(dists, axis=-1).tolist()
    nearest_dists = [dists[idx] for idx in nearest_indices]
    return nearest_indices[:k], nearest_dists[:k]


def dict_to_device(ob, device):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)


def exam_validity(map, path):
    in_collision = False
    for wp in path:
        if map.in_collision(wp[0], wp[1]):
            in_collision = True
            break
    return not in_collision


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
