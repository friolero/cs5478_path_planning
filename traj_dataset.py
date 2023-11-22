import glob
import os.path
import pickle as pkl

import einops
import numpy as np
import torch
from torch.utils.data import Dataset


def flatten(dataset):
    """
    flattens dataset of { key: [ ... x dim ] }
        to { key : [ (...) x dim ]}
    """
    flattened = {}
    for key, xs in dataset.items():
        xs_new = xs
        if xs.ndim == 2:
            # environments (e d)
            pass
        elif xs.ndim == 3:
            # trajectories in fixed environments
            xs_new = einops.rearrange(xs, "b h d -> (b h) d")
        elif xs.ndim == 4:
            # trajectories in variable environments
            xs_new = einops.rearrange(xs, "e b h d -> (e b h) d")
        else:
            raise NotImplementedError
        flattened[key] = xs_new
    return flattened


class Normalizer:
    """
    parent class, subclass by defining the `normalize` and `unnormalize` methods
    """

    def __init__(self, X):
        self.X = X
        self.mins = X.min(dim=0).values
        self.maxs = X.max(dim=0).values

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    -: """
            f"""{torch.round(self.mins, decimals=2)}\n    +: {torch.round(self.maxs, decimals=2)}\n"""
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class LimitsNormalizer(Normalizer):
    """
    maps [ xmin, xmax ] to [ -1, 1 ]
    """

    def normalize(self, x):
        # [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        # [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        """
        x : [ -1, 1 ]
        """
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = torch.clip(x, -1, 1)

        # [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.0

        return x * (self.maxs - self.mins) + self.mins


class DatasetNormalizer:
    def __init__(self, dataset, normalizer):
        dataset = flatten(dataset)

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            self.normalizers[key] = normalizer(val)

    def __repr__(self):
        string = ""
        for key, normalizer in self.normalizers.items():
            string += f"{key}: {normalizer}]\n"
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def get_field_normalizers(self):
        return self.normalizers


class TrajectoryDatasetBase(Dataset):
    def __init__(self, data_dir=None, normalizer="LimitsNormalizer"):

        self.data_dir = data_dir

        self.field_key_traj = "traj"
        self.field_key_task = "task"
        self.fields = {}

        # load data
        self.load_trajectories()

        # dimensions
        b, h, d = self.dataset_shape = self.fields[self.field_key_traj].shape
        self.n_trajs = b
        self.n_support_points = h
        self.state_dim = d  # state dimension used for the diffusion model
        self.trajectory_dim = (self.n_support_points, d)

        # normalize the data (for the diffusion model)
        self.normalizer = DatasetNormalizer(self.fields, normalizer=normalizer)
        self.normalizer_keys = [self.field_key_traj, self.field_key_task]
        self.normalize_all_data(*self.normalizer_keys)

    def load_trajectories(self):
        # load free trajectories
        n_trajs = 0
        trajs_free_l = []
        files = glob.glob(f"{self.data_dir}/*.pkl")
        for fn in files:
            with open(fn, "rb") as fp:
                tmp_traj_data = pkl.load(fp)
            tmp_traj_data = torch.from_numpy(tmp_traj_data).float()
            n_trajs += tmp_traj_data.shape[0]
            trajs_free_l.append(tmp_traj_data)
        trajs_free = torch.cat(trajs_free_l)
        self.fields[self.field_key_traj] = trajs_free

        # task: start and goal state positions [n_trajectories, 2 * state_dim]
        task = torch.cat(
            (trajs_free[..., 0, :], trajs_free[..., -1, :]), dim=-1
        )
        self.fields[self.field_key_task] = task

    def normalize_all_data(self, *keys):
        for key in keys:
            self.fields[f"{key}_normalized"] = self.normalizer(
                self.fields[f"{key}"], key
            )

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, index):
        # Generates one sample of data - one trajectory and tasks
        field_traj_normalized = f"{self.field_key_traj}_normalized"
        field_task_normalized = f"{self.field_key_task}_normalized"
        traj_normalized = self.fields[field_traj_normalized][index]
        task_normalized = self.fields[field_task_normalized][index]
        data = {
            field_traj_normalized: traj_normalized,
            field_task_normalized: task_normalized,
        }

        # build hard conditions
        hard_conds = self.get_hard_conditions(
            traj_normalized, horizon=len(traj_normalized)
        )
        data.update({"hard_conds": hard_conds})

        return data

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        raise NotImplementedError

    def get_unnormalized(self, index):
        raise NotImplementedError
        traj = self.fields[self.field_key_traj][index][..., : self.state_dim]
        task = self.fields[self.field_key_task][index]
        data = {self.field_key_traj: traj, self.field_key_task: task}

        # hard conditions
        # hard_conds = self.get_hard_conds(tasks)
        hard_conds = self.get_hard_conditions(traj)
        data.update({"hard_conds": hard_conds})

        return data

    def unnormalize(self, x, key):
        return self.normalizer.unnormalize(x, key)

    def normalize(self, x, key):
        return self.normalizer.normalize(x, key)

    def unnormalize_trajectories(self, x):
        return self.unnormalize(x, self.field_key_traj)

    def normalize_trajectories(self, x):
        return self.normalize(x, self.field_key_traj)

    def unnormalize_tasks(self, x):
        return self.unnormalize(x, self.field_key_task)

    def normalize_tasks(self, x):
        return self.normalize(x, self.field_key_task)


class TrajectoryDataset(TrajectoryDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        # start and goal positions
        start_state = traj[0][..., : self.state_dim]
        goal_state = traj[-1][..., : self.state_dim]

        if normalize:
            start_state = self.normalizer.normalize(
                start_state, key=self.field_key_traj
            )
            goal_state = self.normalizer.normalize(
                goal_state, key=self.field_key_traj
            )

        if horizon is None:
            horizon = self.n_support_points
        hard_conds = {0: start_state, horizon - 1: goal_state}
        return hard_conds
