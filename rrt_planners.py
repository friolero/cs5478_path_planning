import random

import numpy as np
from base_planner import BasePlanner
from primitives import Node
from utils import distance, knn, sort_with_distance


class RRT(BasePlanner):
    def __init__(self, n_samples, delta_dist):
        self._n_samples = n_samples
        self._delta_dist = delta_dist

    def extend(self, start_node, end_node, map, dist):
        if dist is None:
            dist = distance(start_node, end_node)
        dist_ratio = self._delta_dist / dist
        expand_node = Node(
            start_node.x + int(dist_ratio * (end_node.x - start_node.x)),
            start_node.y + int(dist_ratio * (end_node.y - start_node.y)),
            start_node,
        )
        if not map.in_range(expand_node):
            return None
        elif map.line_in_collision(start_node, expand_node):
            return None
        else:
            return expand_node

    def retrace_path(self, node):
        path = []
        while True:
            path.insert(0, node)
            if node.parent is None:
                break
            else:
                node = node.parent
        return path

    def plan(self, map, start_node, end_node):
        # start from start point
        node_list = [start_node]
        # randomly sample N points
        random_vertices = np.random.choice(
            map.free_conf, size=self._n_samples, replace=True
        )

        path = None
        for i in range(self._n_samples):
            random_vertice = random_vertices[i]
            nearest_idx, nearest_dist = sort_with_distance(
                random_vertice, node_list
            )
            if nearest_dist == 0:
                continue
            expand_node = self.extend(
                node_list[nearest_idx], random_vertice, map, nearest_dist
            )
            if expand_node is None:
                continue
            node_list.append(expand_node)
            if (distance(expand_node, end_node) < self._delta_dist) and (
                not map.line_in_collision(expand_node, end_node)
            ):
                path = self.retrace_path(expand_node)
                path.append(end_node)
                break
            else:
                _, nearest_dist = sort_with_distance(end_node, node_list)
                print(i, len(node_list), nearest_dist)
        return path


class BiRRT(RRT):
    def reverse_parent(self, node_list, root_parent=None):
        reversed_node_list = []
        for i, node in enumerate(node_list):
            if i == 0:
                parent = root_parent
            else:
                parent = node_list[-i]
            reversed_node_list.append(
                Node(node_list[-(i + 1)].x, node_list[-(i + 1)].y, parent)
            )
        return reversed_node_list

    def plan(self, map, start_node, end_node):

        node_lists = [[start_node], [end_node]]
        random_vertices_lists = [
            np.random.choice(map.free_conf, size=self._n_samples, replace=True),
            np.random.choice(map.free_conf, size=self._n_samples, replace=True),
        ]

        path = None
        for i in range(self._n_samples):
            # expand the tree starting from start node
            random_vertice_0 = random_vertices_lists[0][i]
            nearest_idx, nearest_dist = sort_with_distance(
                random_vertice_0, node_lists[0]
            )
            if nearest_dist == 0:
                continue
            expand_node_0 = self.extend(
                node_lists[0][nearest_idx], random_vertice_0, map, nearest_dist
            )
            if expand_node_0 is None:
                continue
            # expand the tree starting from end node
            random_vertice_1 = random_vertices_lists[1][i]
            nearest_idx, nearest_dist = sort_with_distance(
                random_vertice_1, node_lists[1]
            )
            if nearest_dist == 0:
                continue
            expand_node_1 = self.extend(
                node_lists[1][nearest_idx], random_vertice_1, map, nearest_dist
            )
            if expand_node_1 is None:
                continue

            node_lists[0].append(expand_node_0)
            node_lists[1].append(expand_node_1)

            # see if it is possible to connect from  tree 1
            nearest_idx_0, nearest_dist_0 = sort_with_distance(
                expand_node_0, node_lists[1]
            )
            if (nearest_dist_0 < self._delta_dist) and (
                not map.line_in_collision(
                    expand_node_0, node_lists[1][nearest_idx_0]
                )
            ):
                first_path = self.retrace_path(node_lists[0][-1])
                second_path = self.retrace_path(node_lists[1][nearest_idx_0])
                path = first_path + self.reverse_parent(
                    second_path, root_parent=first_path[-1]
                )
                return path

            # see if it is possible to connect from  tree 2
            nearest_idx_1, nearest_dist_1 = sort_with_distance(
                expand_node_1, node_lists[0]
            )
            if (nearest_dist_1 < self._delta_dist) and (
                not map.line_in_collision(
                    expand_node_1, node_lists[0][nearest_idx_1]
                )
            ):
                first_path = self.retrace_path(node_lists[0][nearest_idx_1])
                second_path = self.retrace_path(node_lists[1][-1])
                path = first_path + self.reverse_parent(
                    second_path, root_parent=first_path[-1]
                )
                return path
            print(i, nearest_dist_0, nearest_dist_1)

        return path


class RRTStar(RRT):
    def __init__(self, n_samples, delta_dist, n_neighbors):
        self._n_neighbors = n_neighbors
        super().__init__(n_samples, delta_dist)

    def plan(self, map, start_node, end_node):

        node_list = [start_node]
        node_cost = [0]
        random_vertices = np.random.choice(
            map.free_conf, size=self._n_samples, replace=True
        )

        path = None
        for i in range(self._n_samples):
            random_vertice = random_vertices[i]
            nearest_idx, nearest_dist = sort_with_distance(
                random_vertice, node_list
            )
            if nearest_dist == 0:
                continue
            expand_node = self.extend(
                node_list[nearest_idx], random_vertice, map, nearest_dist
            )
            if expand_node is None:
                continue
            if len(node_list) > self._n_neighbors:
                knn_indices, knn_dists = knn(
                    expand_node, node_list, self._n_neighbors
                )
                tmp_knn_cost = []
                for idx, dist in zip(knn_indices, knn_dists):
                    tmp_knn_cost.append(node_cost[idx] + dist)
                nearest_idx = knn_indices[np.argmin(tmp_knn_cost)]
                expand_node.parent = node_list[nearest_idx]
                node_list.append(expand_node)
                expand_node_cost = min(tmp_knn_cost)
                node_cost.append(expand_node_cost)
                for idx, dist in zip(knn_indices, knn_dists):
                    if (expand_node_cost + dist) < node_cost[idx]:
                        node_list[idx].parent = node_list[-1]
                        node_cost[idx] = expand_node_cost + dist
            else:
                node_list.append(expand_node)
                node_cost.append(node_cost[nearest_idx] + nearest_dist)

            if (distance(expand_node, end_node) < self._delta_dist) and (
                not map.line_in_collision(expand_node, end_node)
            ):
                path = self.retrace_path(expand_node)
                path.append(end_node)
                break
            else:
                _, nearest_dist = sort_with_distance(end_node, node_list)
                print(i, len(node_list), nearest_dist)
        return path
