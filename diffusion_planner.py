import math
import os
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from base_planner import BasePlanner
from diffusion_model import (
    GaussianDiffusion,
    GaussianDiffusionLoss,
    TemporalUnet,
)
from map import ImageMap2D
from traj_dataset import TrajectoryDataset
from utils import dict_to_device, exam_validity, save_vis_paths, set_seed


class DiffusionPlanner(BasePlanner):
    def __init__(
        self,
        n_support_points=64,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
        variance_schedule="exponential",
        n_diffusion_steps=25,
        predict_epsilon=True,
        pretrained_fn="",
        data_dir="",
        device=torch.device("cuda"),
        **kwargs,
    ):

        self._device = device

        self._state_dim = 2  # x, y
        self._n_support_points = n_support_points
        self._unet_input_dim = unet_input_dim
        self._dim_mults = dim_mults
        self._variance_schedule = variance_schedule
        self._n_diffusion_steps = n_diffusion_steps
        self._predict_epsilon = predict_epsilon

        assert os.path.isdir(data_dir), "Please provide a valid data directory."
        self._dataset = TrajectoryDataset(data_dir=data_dir)

        self.unet = TemporalUnet(
            n_support_points=self._n_support_points,
            state_dim=self._state_dim,
            unet_input_dim=self._unet_input_dim,
            dim_mults=self._dim_mults,
        ).to(self._device)

        self.planner = GaussianDiffusion(
            model=self.unet,
            variance_schedule=self._variance_schedule,
            n_diffusion_steps=self._n_diffusion_steps,
            predict_epsilon=self._predict_epsilon,
        ).to(self._device)

        self._pretrained_fn = pretrained_fn
        if os.path.isfile(self._pretrained_fn):
            self.load(self._pretrained_fn)

    def calculate_path_metrics(self, path):
        def cal_curvature(path):
            data = np.vstack([(wp[0], wp[1]) for wp in path])

            # first derivatives
            dx = np.gradient(data[:, 0])
            dy = np.gradient(data[:, 1])

            # second derivatives
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)

            # calculation of curvature from the typical formula
            curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5
            return np.nan_to_num(curvature, nan=0.0)

        path_length = sum(
            [
                np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
                for i in range(len(path) - 1)
            ]
        )
        curvature = cal_curvature(path)
        mean_curvature = curvature.mean()
        max_curvature = np.abs(curvature).max()

        return path_length, mean_curvature, max_curvature

    @property
    def dataset(self):
        return self._dataset

    def load(self, ckpt_fn):
        print(f"==> Loaded pretrained checkpoint from {ckpt_fn}.")
        self.planner.model.load_state_dict(
            torch.load(ckpt_fn).model.to(self._device).state_dict()
        )

    def save(self, pretrained_fn):
        print(f"==> Saving pretrained checkpoint to {ckpt_fn}...")
        torch.save(self.planner, self._pretrained_fn)

    def get_hard_conds(self, start_node, end_node):
        traj = torch.Tensor(
            [[start_node.x, start_node.y], [end_node.x, end_node.y]]
        ).float()
        hard_conds = self._dataset.get_hard_conditions(traj, normalize=True)
        return hard_conds

    def plan(
        self,
        map,
        start_node,
        end_node,
        n_samples,
        unnormalize=True,
        **sample_fn_kwargs,
    ):
        hard_conds = self.get_hard_conds(start_node, end_node)
        with torch.no_grad():
            self.planner.eval()
            paths = self.planner.run_inference(
                context=None,
                hard_conds=hard_conds,
                n_samples=n_samples,
                horizon=self._dataset.n_support_points,
                return_chain=False,
                **sample_fn_kwargs,
                n_diffusion_steps_without_noises=5,
            ).cpu()
            if unnormalize:
                paths = self._dataset.unnormalize_trajectories(paths)
            paths = [path.astype(int).tolist() for path in paths.numpy()]
        return paths

    def guided_plan(
        self,
        map,
        start_node,
        end_node,
        guide,
        n_samples,
        unnormalize=True,
        n_guide_steps=5,
        start_guide_steps_fraction=0.25,
        n_diffusion_steps_without_noises=5,
    ):
        t_start_guide = math.ceil(
            start_guide_steps_fraction * self._n_diffusion_steps
        )
        sample_fn_kwargs = dict(
            guide=guide,
            n_guide_steps=n_guide_steps,
            t_start_guide=t_start_guide,
            noise_std_extra_schedule_fn=lambda x: 0.5,
        )
        hard_conds = self.get_hard_conds(start_node, end_node)
        with torch.no_grad():
            self.planner.eval()
            paths = self.planner.run_inference(
                context=None,
                hard_conds=hard_conds,
                n_samples=n_samples,
                horizon=self._dataset.n_support_points,
                return_chain=False,
                **sample_fn_kwargs,
                n_diffusion_steps_without_noises=n_diffusion_steps_without_noises,
            ).cpu()
            if unnormalize:
                paths = self._dataset.unnormalize_trajectories(paths)
            paths = [path.astype(int).tolist() for path in paths.numpy()]
        return paths

    def train(
        self, lr=3e-4, batch_size=128, num_train_steps=500000, max_grad_norm=1.0
    ):

        train_dataset, val_dataset = random_split(self._dataset, [0.9, 0.1])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = GaussianDiffusionLoss().loss_fn
        optimizer = torch.optim.Adam(lr=lr, params=self.planner.parameters())

        num_epochs = int(num_train_steps * batch_size / len(train_dataset))
        for epoch in range(num_epochs):
            self.planner.train()
            for idx, batch in enumerate(train_dataloader):
                batch = dict_to_device(batch, device)
                train_loss_dict, train_loss_info = criterion(
                    self.planner, batch, train_dataset.dataset
                )
                train_loss_value = 0.0
                print_str = f"Train batch [{idx}]: "
                for name, loss in train_loss_dict.items():
                    single_loss = loss.mean()
                    train_loss_value += single_loss
                    print_str += f"{name} - {single_loss}, "
                print(print_str[:-2])

                optimizer.zero_grad()
                train_loss_value.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.planner.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()

            self.planner.eval()
            for idx, batch in enumerate(val_dataloader):
                batch = dict_to_device(batch, device)
                with torch.no_grad():
                    val_loss_dict, val_loss_info = criterion(
                        self.planner, batch, val_dataset.dataset
                    )
                    val_loss_value = 0.0
                    print_str = f"Eval batch [{idx}]: "
                    for name, loss in val_loss_dict.items():
                        single_loss = loss.mean()
                        val_loss_value += single_loss
                        print_str += f"{name} - {single_loss}, "
                print(print_str[:-2])

    def test(
        self, map, n_test, n_samples=5, save_dir="vis/diffusion", fn_prefix=""
    ):
        if save_dir is not None:
            save_vis = True
            if not os.path.isdir(save_dir):
                os.system(f"mkdir -p {save_dir}")
        total_paths = []
        total_valid_paths = []
        for test_idx in range(n_test):
            start_node, end_node = np.random.choice(map.free_conf, size=2)
            paths = self.plan(map, start_node, end_node, n_samples=n_samples)
            valid_paths = [exam_validity(map, path) for path in paths]
            if save_vis:
                out_fn = f"{save_dir}/{fn_prefix}_{test_idx:02d}.png"
                save_vis_paths(map, paths, out_fn)
            input(
                f"[{test_idx}]: start position [{paths[0][0][0]}, {paths[0][0][1]}], end position [{paths[0][-1][0]}, {paths[0][-1][1]}], {sum(valid_paths)} / {len(valid_paths)} valid path planned.\nPress enter to continue."
            )
            total_paths.append(paths)
            total_valid_paths.append(valid_paths)
        return total_paths, total_valid_paths


class CollisionCostGuide:
    def __init__(
        self,
        map,
        dataset,
        smooth_guide_weight=0.1,
        col_guide_weight=0.0,
        tol_radius=5,
        vis=False,
    ):
        self._map = map
        self._dataset = dataset
        self._tol_radius = tol_radius
        self._smooth_guide_weight = smooth_guide_weight
        self._col_guide_weight = col_guide_weight
        self._map.build_map_cost_grad(tol_radius=self._tol_radius, vis=vis)

    def grid_cost_grad(self, wp):
        wp = wp.clone().cpu().numpy().astype(int)
        col_cost_grad = torch.from_numpy(self._map.col_cost_grad[wp[0], wp[1]])
        col_cost_grad = col_cost_grad.float().unsqueeze(0).unsqueeze(1)
        return (-1) * col_cost_grad

    def line_cost_grad(self, wp_1, wp_2):
        x1, y1 = wp_1.clone().cpu().numpy().astype(int).tolist()
        x2, y2 = wp_2.clone().cpu().numpy().astype(int).tolist()

        n_check = max(abs(x2 - x1), abs(y2 - y1))
        if n_check == 0:
            col_cost_grad = self._map.col_cost_grad[x1, y1]
        else:
            col_cost_grad = np.zeros(2)
            step_size = (float(x2 - x1) / n_check, float(y2 - y1) / n_check)
            for i in range(n_check):
                tmp_x = int(x1 + (i + 1) * step_size[0])
                tmp_y = int(y1 + (i + 1) * step_size[1])
                col_cost_grad += self._map.col_cost_grad[tmp_x, tmp_y]
            col_cost_grad /= n_check
        col_cost_grad = torch.from_numpy(col_cost_grad)
        col_cost_grad = col_cost_grad.float().unsqueeze(0).unsqueeze(1)
        return (-1) * col_cost_grad

    def smoothness_cost_grad(self, x):
        pos_diff = torch.diff(x, dim=1)
        return pos_diff

    def __call__(self, x, interpolate=False):
        unnormalized_x = self._dataset.unnormalize_trajectories(x.clone().cpu())
        unnormalized_x[:, :, 0] = torch.clip(
            unnormalized_x[:, :, 0], min=0, max=self._map.row - 1
        )
        unnormalized_x[:, :, 1] = torch.clip(
            unnormalized_x[:, :, 1], min=0, max=self._map.col - 1
        )
        col_cost_grads = []
        for i in range(x.shape[0]):
            if interpolate:
                col_cost_grads.append(
                    torch.cat(
                        [
                            self.line_cost_grad(
                                unnormalized_x[0][i],
                                unnormalized_x[0][min(x.shape[1] - 1, i + 1)],
                            )
                            for i in range(x.shape[1])
                        ],
                        1,
                    )
                )
            else:
                col_cost_grads.append(
                    torch.cat(
                        [
                            self.grid_cost_grad(unnormalized_x[0][i])
                            for i in range(x.shape[1])
                        ],
                        1,
                    )
                )
            col_cost_grads[-1][0][0].fill_(0)
            col_cost_grads[-1][0][-1].fill_(0)
        col_cost_grads = torch.cat(col_cost_grads, 0).to(x.device)
        # col_cost_grads = torch.zeros_like(col_cost_grads)
        smooth_cost_grads = torch.zeros_like(col_cost_grads)
        smooth_cost_grads[:, 1:-1, :] = self.smoothness_cost_grad(x)[:, 1:, :]
        return (
            self._col_guide_weight * col_cost_grads
            + self._smooth_guide_weight * smooth_cost_grads
        )


if __name__ == "__main__":

    map_fn = "data/2d_maze_2.png"
    # map_fn = "data/2d_map_4.png"
    prefix = f"{map_fn.split('/')[-1].split('.')[0]}"
    data_dir = f"data/traj_data/{prefix}_BiRRT"
    ckpt_fn = f"data/{prefix}_diffusion.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = DiffusionPlanner(
        pretrained_fn=ckpt_fn, data_dir=data_dir, device=device
    )

    if not os.path.isfile(ckpt_fn):
        planner.train()
        planner.save(ckpt_fn)
    else:
        planner.load(ckpt_fn)

    with open("data/eval_cases.pkl", "rb") as fp:
        eval_cases = pkl.load(fp)
    map = ImageMap2D(map_fn)

    guide = CollisionCostGuide(map, planner.dataset)
    # planner.test(map, n_test=50, fn_prefix="final")
    for i, (start_node, end_node) in enumerate(eval_cases):
        print(f"     ==> test case {i}")

        unguided_start_time = time.time()
        paths = planner.plan(
            map, start_node, end_node, unnormalize=True, n_samples=5
        )
        unguided_end_time = time.time()
        unguided_time = unguided_end_time - unguided_start_time

        valid_paths = [exam_validity(map, path) for path in paths]
        save_vis_paths(map, paths, out_fn=f"diffusion_unguided_{i}.png")
        # input("unguided path")
        for path in paths:
            path_length, mean_curvature, max_curvature = planner.calculate_path_metrics(
                path
            )
            print(
                f"        Success: {sum(valid_paths)}; path length: {path_length}; mean curvature: {mean_curvature}; Time taken: {unguided_time}s"
            )

        """
        guide._smooth_guide_weight = 0.0
        guide._col_guide_weight = 0.08
        guided_start_time = time.time()
        guided_paths = planner.guided_plan(
            map, start_node, end_node, guide, unnormalize=True, n_samples=1
        )
        guided_end_time = time.time()
        guided_time = guided_end_time - guided_start_time

        valid_paths = [exam_validity(map, path) for path in guided_paths]
        save_vis_paths(
            map, guided_paths, out_fn="diffusion_collision_guided.png"
        )
        input("Collision guided path")
        for path in guided_paths:
            path_length, mean_curvature, max_curvature = planner.calculate_path_metrics(
                path
            )
            print(
                f"        Success: {sum(valid_paths)}; path length: {path_length}; mean curvature: {mean_curvature}; Time taken: {guided_time}s"
            )

        guide._smooth_guide_weight = 0.25
        guide._col_guide_weight = 0.0
        guided_start_time = time.time()
        guided_paths = planner.guided_plan(
            map, start_node, end_node, guide, unnormalize=True, n_samples=1
        )
        guided_end_time = time.time()
        guided_time = guided_end_time - guided_start_time

        valid_paths = [exam_validity(map, path) for path in guided_paths]
        save_vis_paths(
            map, guided_paths, out_fn="diffusion_smoothness_guided.png"
        )
        input("Smoothness guided path")
        for path in guided_paths:
            path_length, mean_curvature, max_curvature = planner.calculate_path_metrics(
                path
            )
            print(
                f"        Success: {sum(valid_paths)}; path length: {path_length}; mean curvature: {mean_curvature}; Time taken: {guided_time}s"
            )

        guide._smooth_guide_weight = 0.25
        guide._col_guide_weight = 0.08
        guided_start_time = time.time()
        guided_paths = planner.guided_plan(
            map, start_node, end_node, guide, unnormalize=True, n_samples=5
        )
        guided_end_time = time.time()
        guided_time = guided_end_time - guided_start_time

        valid_paths = [exam_validity(map, path) for path in guided_paths]
        save_vis_paths(
            map, guided_paths, out_fn=f"diffusion_combined_guided_{i}.png"
        )
        # input("Combined guided path")
        for path in guided_paths:
            path_length, mean_curvature, max_curvature = planner.calculate_path_metrics(
                path
            )
            print(
                f"        Success: {sum(valid_paths)}; path length: {path_length}; mean curvature: {mean_curvature}; Time taken: {guided_time}s"
            )
        """
