import abc
import os.path

import git
import numpy as np
import torch
from torch.utils.data import Dataset

from normalizer import DatasetNormalizer


class TrajectoryDatasetBase(Dataset):
    def __init__(self, base_dir=None, normalizer="LimitsNormalizer"):

        # self._base_dir

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
        # TODO load free trajectories
        # load trajs_free_l, n_trajs
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
