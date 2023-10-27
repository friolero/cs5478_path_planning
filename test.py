import sys
import time

import numpy as np
from map import ImageMap2D
from rrt_planners import RRT, BiRRT, RRTStar
from utils import distance, vis_map, vis_path

if __name__ == "__main__":

    map = ImageMap2D(sys.argv[1])
    # vis_map(map)

    feasible_conf = False
    while not feasible_conf:
        start_node, end_node = np.random.choice(
            map.free_conf, size=2, replace=False
        )
        dist = distance(start_node, end_node)
        print(f"==> Start node: ({start_node.x}, {start_node.y})")
        print(f"==> End node: ({end_node.x}, {end_node.y})")
        print(f"==> Distance: {dist}")
        vis_path(map, [start_node, end_node])
        feasible_conf = input("feasible configuration to plan? (y/n)") in [
            "y",
            "Y",
        ]

    n_samples = 100000
    planners = [
        ("RRT", RRT(n_samples)),
        ("BiRRT", BiRRT(n_samples)),
        ("RRT*", RRTStar(n_samples, n_neighbors=5)),
    ]

    for (name, planner) in planners:
        print(f"==> Using planner: {name}")
        start_time = time.time()
        path = planner.plan(map, start_node, end_node)
        end_time = time.time()
        if path is None:
            print(f"    Fail to find a path with {n_samples} samples.")
        else:
            print(
                f"    Path length: {len(path)}; Time taken: {end_time - start_time}s."
            )
            vis_path(map, path)
