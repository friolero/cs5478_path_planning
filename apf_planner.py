import numpy as np
from primitives import Node
from utils import distance


class ArtificialPotentialField:
    def __init__(
        self,
        k_att=1.0,
        k_rep=1,
        max_iterations=1000,
        step_size=5.0,
        radius=30,
        auto_tune=False,
    ):
        self._k_att = k_att
        self._k_rep = k_rep
        self._max_iterations = max_iterations
        self._step_size = step_size
        self._r = radius
        self._auto_tune = auto_tune

    def attraction(self, curr_node, end_node):
        att_force = np.array(
            [end_node.x - curr_node.x, end_node.y - curr_node.y],
            dtype=np.float32,
        )
        if (att_force != 0).any():
            att_force /= np.linalg.norm(att_force)
        return att_force

    def repulsion(self, map, curr_node, eps=1e-6):
        rep_force = np.array([0, 0], dtype=np.float32)
        for obs_node in map.obstacle_conf:
            dist = max(distance(curr_node, obs_node), eps)
            if dist <= self._r:
                rep_force += (
                    self._k_rep
                    * np.array(
                        [curr_node.x - obs_node.x, curr_node.y - obs_node.y],
                        dtype=np.float32,
                    )
                    * (1.0 / dist - 1.0 / self._r)
                    / dist
                )
        if (rep_force != 0).any():
            rep_force /= np.linalg.norm(rep_force)
        return rep_force

    def plan(self, map, start_node, end_node):
        path = [start_node]
        curr_node = start_node

        for i in range(self._max_iterations):

            att_ptl = self._k_att * self.attraction(curr_node, end_node)
            rep_ptl = self._k_rep * self.repulsion(map, curr_node)
            if self._auto_tune:
                expand_coef = 1.0
                while True:
                    total_force = expand_coef * att_ptl + rep_ptl
                    valid = (int(total_force[0] * self._step_size) != 0) or (
                        int(total_force[1] * self._step_size) != 0
                    )
                    if not valid:
                        expand_coef += 0.1
                    else:
                        break
            else:
                total_force = att_ptl + rep_ptl

            delta_x = int(total_force[0] * self._step_size)
            delta_y = int(total_force[1] * self._step_size)
            expand_node = Node(
                curr_node.x + int(delta_x),
                curr_node.y + int(delta_y),
                curr_node,
            )
            path.append(expand_node)
            if distance(expand_node, end_node) <= self._step_size:
                return path + [end_node], True
            else:
                curr_node = path[-1]
                # print(i, delta_x, delta_y, distance(curr_node, end_node))
        return path + [end_node], False
