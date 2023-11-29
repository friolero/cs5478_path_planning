import math
import os

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
from utils import dict_to_device, exam_validity, save_vis_paths


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
        data_dir="data/traj_data/BiRRT",
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

    @property
    def dataset(self):
        return self._dataset

    def load(self, ckpt_fn):
        print(
            f"==> Loaded pretrained checkpoint from {ckpt_fn}. Start testing..."
        )
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
    def __init__(self, mpa, dataset, guide_weight=1, tol_radius=5):
        self._map = map
        self._dataset = dataset
        self._tol_radius = tol_radius
        self._guide_weight = guide_weight

    def d2c(self, dist):
        return 1.0 / (2 * self._tol_radius) * (dist - self._tol_radius) ** 2

    def potential_grad(self, dist, x, y):
        if x < map.row - 1:
            grad_x = self.d2c(
                self._map.nearest_obstacle(x + 1, y)[0]
            ) - self.d2c(dist)
        else:
            grad_x = self.d2c(dist) - self.d2c(
                self._map.nearest_obstacle(x - 1, y)[0]
            )
        if y < self._map.col - 1:
            grad_y = self.d2c(
                self._map.nearest_obstacle(x, y + 1)[0]
            ) - self.d2c(dist)
        else:
            grad_y = self.d2c(dist) - self.d2c(
                self._map.nearest_obstacle(x, y - 1)[0]
            )
        return np.array([grad_x, grad_y], dtype=np.float32)

    def grid_cost_grad(self, wp):
        wp = wp.clone().cpu().numpy().astype(int)
        dist, delta_cost = self._map.nearest_obstacle(wp[0], wp[1])
        if dist <= self._tol_radius:
            cost_grad = self.potential_grad(dist, wp[0], wp[1])
        else:
            cost_grad = np.zeros(2)
        cost_grad = (
            torch.from_numpy(cost_grad).float().unsqueeze(0).unsqueeze(1)
        )
        cost_grad.fill_(0)
        return (-1) * cost_grad

    def __call__(self, x):
        unnormalized_x = self._dataset.unnormalize_trajectories(x.clone().cpu())
        traj_cost_grads = []
        for i in range(x.shape[0]):
            traj_cost_grads.append(
                torch.cat(
                    [
                        self.grid_cost_grad(unnormalized_x[0][i])
                        for i in range(x.shape[1])
                    ],
                    1,
                )
            )
            traj_cost_grads[-1][0][0].fill_(0)
            traj_cost_grads[-1][0][-1].fill_(0)
        traj_cost_grads = torch.cat(traj_cost_grads, 0).to(x.device)
        return self._guide_weight * traj_cost_grads


if __name__ == "__main__":

    ckpt_fn = "data/diffusion.pt"
    data_dir = "data/traj_data/BiRRT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = DiffusionPlanner(
        pretrained_fn=ckpt_fn, data_dir=data_dir, device=device
    )

    if not os.path.isfile(ckpt_fn):
        planner.train()
        planner.save(ckpt_fn)
    else:
        planner.load(ckpt_fn)

    map = ImageMap2D("data/2d_maze_2.png")
    # planner.test(map, n_test=50, fn_prefix="final")
    for test_idx in range(50):
        start_node, end_node = np.random.choice(map.free_conf, size=2)
        guide = CollisionCostGuide(map, planner.dataset)
        paths = planner.guided_plan(
            map, start_node, end_node, guide, unnormalize=True, n_samples=5
        )
        valid_paths = [exam_validity(map, path) for path in paths]
        save_vis_paths(map, paths, out_fn=None)
        input(
            f"[{test_idx}]: start position [{paths[0][0][0]}, {paths[0][0][1]}], end position [{paths[0][-1][0]}, {paths[0][-1][1]}], {sum(valid_paths)} / {len(valid_paths)} valid path planned.\nPress enter to continue."
        )
