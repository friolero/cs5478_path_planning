import random

import numpy as np
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def vis_map(map):
    image = Image.fromarray((map.map * 255.0).astype(np.uint8))
    image.show()


def distance(point_a, point_b):
    return np.linalg.norm([point_a.x - point_b.x, point_a.y - point_b.y])


def sort_with_distance(node, neighbor_nodes):
    dists = [distance(node, neighbor_node) for neighbor_node in neighbor_nodes]
    dists = np.array(dists).reshape(-1)
    nearest_indices = np.argmin(dists, axis=-1)
    nearest_dists = [dists[idx] for idx in nearest_indices]
    return nearest_indices[0], nearest_dists[0]


# def neighbor_in_range
