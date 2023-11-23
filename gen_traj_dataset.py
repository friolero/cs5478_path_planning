import argparse
import os
import pickle as pkl
import sys
import time
from multiprocessing import Process, Queue

import numpy as np
from tqdm import tqdm

from map import ImageMap2D
from rrt_planners import RRT, BiRRT, RRTStar
from utils import distance, save_vis_paths, set_seed

parser = argparse.ArgumentParser(description="Trajectory generation.")
parser.add_argument("-seed", type=int, default=77, help="random seed")
parser.add_argument(
    "-out_dir", type=str, default="data/traj_data", help="output directory"
)
parser.add_argument(
    "-n_parallel", type=int, default=20, help="number of parallel processes"
)
parser.add_argument(
    "-map_fn", type=str, default="data/2d_maze_2.png", help="2d map filename"
)
parser.add_argument(
    "-planner", type=str, default="BiRRT", choices=["RRT", "BiRRRT", "RRTStar"]
)
parser.add_argument(
    "-n_samples", type=int, default=20000, help="sample numbers for planner"
)
parser.add_argument(
    "-n_task", type=int, default=500, help="total context number"
)
parser.add_argument(
    "-max_plan_time", type=int, default=80, help="maximum secs to plan a task"
)
parser.add_argument(
    "-n_traj_per_task", type=int, default=20, help="trajectory per context"
)
args = parser.parse_args()


class TrajectoryGenerator:
    def __init__(
        self,
        map,
        out_dir,
        n_task,
        n_traj_per_task,
        n_parallel,
        max_plan_time,
        seed,
    ):
        self._seed = seed
        set_seed(self._seed)

        self._out_dir = out_dir
        if not os.path.isdir(self._out_dir):
            print(f"*****Creating folder {self._out_dir}*****")
            os.system(f"mkdir -p {self._out_dir}")
        else:
            res = input(f"*****{self._out_dir} existed. Overwrite (y/n)?*****")
            if res.lower() == "y":
                os.system(f"rm -rf {self._out_dir}")
                os.system(f"mkdir -p {self._out_dir}")

        self._map = map
        self._n_wps = 64
        self._n_task = n_task
        self._n_traj_per_task = n_traj_per_task
        self._tasks = [self.gen_random_configs() for _ in range(n_task)]

        self._n_parallel = n_parallel
        self._max_plan_time = max_plan_time

    def gen_random_configs(self, min_dist=100):
        valid = False
        while not valid:
            start_node, end_node = np.random.choice(
                self._map.free_conf, size=2, replace=False
            )
            valid = distance(start_node, end_node) > min_dist
        return (start_node, end_node)

    def postprocess(self, trajectory):
        trajectory = np.array([np.array([wp.x, wp.y]) for wp in trajectory])
        idx = np.round(np.linspace(0, len(trajectory) - 1, self._n_wps)).astype(
            int
        )
        trajectory = trajectory[idx][np.newaxis, ...]
        return trajectory

    def plan_single_config(self, planner, task_config, queue, config_idx):
        start_node, end_node = task_config
        task_trajs = []
        start_time = time.time()
        while len(task_trajs) < self._n_traj_per_task:
            path, success = planner.plan(self._map, start_node, end_node)
            if time.time() - start_time > self._max_plan_time:
                break
            if not success:
                continue
            elif len(path) < self._n_wps:
                # print("Path too short!", len(path))
                continue
            else:
                path = self.postprocess(path)
            if success:
                task_trajs.append(path)
        end_time = time.time()
        print(f"[Task {config_idx}]: time taken: {end_time - start_time}s.")
        if len(task_trajs) > 0:
            task_trajs = np.vstack(task_trajs)
        else:
            task_trajs = np.empty((0, self._n_wps, 2))
        queue.put((config_idx, task_trajs))

    def __call__(self, planner):

        queue = Queue()
        if self._n_parallel > 1:
            for config_start_idx in range(0, self._n_task, self._n_parallel):
                config_end_idx = min(
                    self._n_task - 1, config_start_idx + self._n_parallel
                )
                print(
                    f"==> Generating trajectory {config_start_idx}-{config_end_idx}"
                )
                procs = []
                for config_idx in range(config_start_idx, config_end_idx):
                    procs.append(
                        Process(
                            target=self.plan_single_config,
                            args=(
                                planner,
                                self._tasks[config_idx],
                                queue,
                                config_idx,
                            ),
                        )
                    )
                for proc in procs:
                    proc.start()
                for proc in procs:
                    config_idx, task_trajs = queue.get()
                    if task_trajs.shape[0] < 1:
                        continue
                    save_vis_paths(
                        self._map,
                        task_trajs,
                        f"{self._out_dir}/task_{config_idx:04d}.png",
                    )
                    with open(
                        f"{self._out_dir}/task_{config_idx:04d}.pkl", "wb"
                    ) as fp:
                        pkl.dump(task_trajs, fp)
                for proc in procs:
                    proc.join()
                print(f"==> Done!")
        else:
            for config_idx in range(self._n_task):
                print(f"==> Generating trajectory {config_idx}")
                self.plan_single_config(
                    planner, self._tasks[config_idx], queue, config_idx
                )
                config_idx, task_trajs = queue.get()
                with open(
                    f"{self._out_dir}/task_{config_idx:04d}.pkl", "wb"
                ) as fp:
                    pkl.dump(task_trajs, fp)
                print(f"==> Done!")

        return trajs


if __name__ == "__main__":

    map = ImageMap2D(args.map_fn)
    delta_dist = int(min(map.row, map.col) / 50)
    planners = {
        "RRT": RRT(args.n_samples, delta_dist),
        "BiRRT": BiRRT(args.n_samples, delta_dist),
        "RRTStar": RRTStar(args.n_samples, delta_dist, n_neighbors=5),
    }
    generator = TrajectoryGenerator(
        map,
        out_dir=f"{args.out_dir}/{args.planner}",
        n_task=args.n_task,
        n_traj_per_task=args.n_traj_per_task,
        n_parallel=args.n_parallel,
        max_plan_time=args.max_plan_time,
        seed=args.seed,
    )

    trajs = generator(planners[args.planner])
