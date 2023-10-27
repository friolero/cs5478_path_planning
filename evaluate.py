import sys
import time

import numpy as np
from map import ImageMap2D
from rrt_planners import RRT, BiRRT, RRTStar
from tqdm import tqdm
from utils import distance, set_seed, vis_path


class Evaluator:
    def __init__(self, map, n_eval, seed):
        self._map = map
        self._n_eval = n_eval
        self._seed = seed
        set_seed(self._seed)

        self._eval_cases = [self.gen_random_configs() for _ in range(n_eval)]

    def gen_random_configs(self):
        start_node, end_node = np.random.choice(
            self._map.free_conf, size=2, replace=False
        )
        return (start_node, end_node)

    def __call__(self, planner, vis=False):
        def cal_curvature(path):
            data = np.vstack([(wp.x, wp.y) for wp in path])

            # first derivatives
            dx = np.gradient(data[:, 0])
            dy = np.gradient(data[:, 1])

            # second derivatives
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)

            # calculation of curvature from the typical formula
            curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5
            return curvature

        n_success = 0
        path_lengths = []
        mean_curvatures = []
        max_curvatures = []
        time_taken = []
        for i, (start_node, end_node) in enumerate(self._eval_cases):
            print(f"     ==> test case {i}")
            start_time = time.time()
            path = planner.plan(self._map, start_node, end_node)
            end_time = time.time()

            success = path is not None
            if success:
                n_success += 1
                path_lengths.append(len(path))
                curvature = cal_curvature(path)
                mean_curvatures.append(curvature.mean())
                max_curvatures.append(np.abs(curvature).max())
                if vis:
                    vis_path(self._map, path)
            time_taken.append(end_time - start_time)
            print(
                f"        Success: {success}; path length: {path_lengths[-1]}; mean curvature: {mean_curvatures[-1]}; Time taken: {time_taken[-1]}s"
            )

        results = {
            "success_rate": float(n_success) / len(self._eval_cases),
            "avg_path_length": sum(path_lengths) / len(path_lengths),
            "avg_curvature": sum(mean_curvatures) / len(mean_curvatures),
            "max_curvature": sum(max_curvatures) / len(max_curvatures),
            "time_taken": sum(time_taken) / len(time_taken),
        }
        return results


# RRT         Path length: 144; Time taken: 1125.7872867584229s.
# RRT-star    Path length: 98; Time taken: 357.20960211753845s.


if __name__ == "__main__":

    map = ImageMap2D("data/2d_maze_2.png")
    evaluator = Evaluator(map, n_eval=10, seed=77)

    n_samples = 20000
    delta_dist = int(min(map.row, map.col) / 50)
    planners = {
        "RRT": RRT(n_samples, delta_dist),
        "BiRRT": BiRRT(n_samples, delta_dist),
        "RRTStar": RRTStar(n_samples, delta_dist, n_neighbors=5),
    }

    results = evaluator(planners[sys.argv[1]])
    print("Overall results:")
    print(results)
