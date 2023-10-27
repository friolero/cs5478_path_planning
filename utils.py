import random
from copy import deepcopy

from PIL import Image, ImageDraw

import numpy as np


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
