import numpy as np

from base_planner import BasePlanner
from utils import distance, sort_with_distance


class RRT(BasePlanner):
    def __init__(self, num_vertices, delta_dist):
        self._num_vertices = num_vertices
        self._delta_dist = delta_dist

    def plan(self, map, start_conf, end_conf):
        # randomly sample N points
        # start from start point
        # find nearest point without collision from the current point
        node_list = [start_conf]
        random_vertices = np.random.choice(
            map.free_conf,
            size=min(self._num_vertices, len(map.free_conf)),
            replace=False,
        )
        for i in range(self._num_vertices):
            nearest_idx, nearest_dist = sort_with_distance(
                random_vertices[i], node_list
            )
            if nearest_dist == 0:
                continue
            nearest_node = node_list[nearest_idx]
            dist_ratio = self._delta_dist / nearest_dist
            n_grid = max(
                int(dist_ratio * (random_vertices[i].x - nearest_node.x)),
                int(dist_ratio * (random_vertices[i].y - nearest_node.y)),
            )
