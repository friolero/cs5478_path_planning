# 这玩意看样子可以扔了
import numpy as np

from base_planner import BasePlanner
from primitives import Node
from utils import distance, sort_with_distance


class RPMPlanner(BasePlanner):
    def __init__(self, num_vertices, delta_dist, rewiring_radius):
        self._num_vertices = num_vertices
        self._delta_dist = delta_dist
        self._rewiring_radius = rewiring_radius

    def plan(self, map, start_conf, end_conf):
        # Initialize the tree with the start configuration
        node_list = [start_conf]

        for i in range(self._num_vertices):
            # Sample a random point within the map
            random_point = map.free_conf[np.random.choice(len(map.free_conf))]

            # Find the nearest node in the tree
            nearest_idx, nearest_dist = sort_with_distance(random_point, node_list)
            if nearest_dist == 0:
                continue
            nearest_node = node_list[nearest_idx]

            # Calculate a new node (n_grid) based on the specified delta distance
            dist_ratio = self._delta_dist / nearest_dist
            n_grid_x = int(dist_ratio * (random_point.x - nearest_node.x))
            n_grid_y = int(dist_ratio * (random_point.y - nearest_node.y))
            n_grid = Node(nearest_node.x + n_grid_x, nearest_node.y + n_grid_y)

            # Check if the new node is valid and not in collision
            if not map.is_valid_conf(n_grid):
                continue

            # Perform rewiring: Check if any node in the tree can be reached with a shorter path via the new node
            for j, node in enumerate(node_list):
                if node == nearest_node:
                    continue
                if distance(node, n_grid) < self._rewiring_radius:
                    if map.is_valid_conf(n_grid) and distance(start_conf, n_grid) + distance(n_grid, end_conf) < distance(start_conf, node) + distance(node, end_conf):
                        node_list[j] = n_grid

            # Add the new node to the tree
            node_list.append(n_grid)
