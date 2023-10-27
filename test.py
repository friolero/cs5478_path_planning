import sys

import ipdb
import numpy as np
from map import ImageMap2D
from rrt_planners import RRT, BiRRT, RRTStar
from tqdm import tqdm
from utils import distance, vis_map, vis_path

if __name__ == "__main__":

    map = ImageMap2D(sys.argv[1])
    # vis_map(map)

    start_node, end_node = np.random.choice(
        map.free_conf, size=2, replace=False
    )
    dist = distance(start_node, end_node)
    print(f"==> Start node: ({start_node.x}, {start_node.y})")
    print(f"==> End node: ({end_node.x}, {end_node.y})")
    print(f"==> Distance: {dist}")

    planner = RRT(100000, 5)
    planner = BiRRT(100000, 5)
    planner = RRTStar(100000, 5, n_neighbors=5)

    path = planner.plan(map, start_node, end_node)
    if path is None:
        print("Fail to find a path")
    else:
        vis_path(map, path)
    ipdb.set_trace()
