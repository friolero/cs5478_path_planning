import collections
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from diffusion_model import (
    GaussianDiffusion,
    GaussianDiffusionLoss,
    TemporalUnet,
)
from map import ImageMap2D
from traj_dataset import TrajectoryDataset
from utils import save_vis_paths


# I haven't modified the code
# But I added some comments where it might be possible to optimize


# Move dictionary items 2 a specified device
def dict_to_device(ob, device):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)


def exam_validity(map, path):
    in_collision = False
    for wp in path:
        if map.in_collision(wp[0], wp[1]):
            in_collision = True
            break
    return not in_collision


def test(planner, map, dataset, out_fn, n_samples=5):
    start_node, end_node = np.random.choice(map.free_conf, size=2)
    paths = planner.plan(
        map, start_node, end_node, dataset, n_samples=n_samples
    )
    save_vis_paths(map, paths, out_fn)
    valid_paths = [exam_validity(map, path) for path in paths]
    return paths, valid_paths

def train(
    planner,
    dataset,
    criterion,
    optimizer,
    batch_size,
    num_train_steps,
    device,
    n_test,
    map,
    vis_out_dir,
):

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    num_epochs = int(num_train_steps * batch_size / len(train_dataset))

    # Loop through epochs
    for epoch in range(num_epochs):
        planner.train() # Training mode planner
        for idx, batch in enumerate(train_dataloader):
            batch = dict_to_device(batch, device)
            train_loss_dict, train_loss_info = criterion(
                planner, batch, train_dataset.dataset
            )
            train_loss_value = 0.0
            print_str = f"Train batch [{idx}]: "
            for name, loss in train_loss_dict.items():
                single_loss = loss.mean()
                train_loss_value += single_loss
                print_str += f"{name} - {single_loss}, "
            print(print_str[:-2])

            # Backpropagation and optimization
            optimizer.zero_grad()
            train_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), max_norm=1.0)
            optimizer.step()

        planner.eval()
        for idx, batch in enumerate(val_dataloader):
            batch = dict_to_device(batch, device)
            with torch.no_grad():
                val_loss_dict, val_loss_info = criterion(
                    planner, batch, val_dataset.dataset
                )
                val_loss_value = 0.0
                print_str = f"Eval batch [{idx}]: "

                # Validation losses
                for name, loss in val_loss_dict.items():
                    single_loss = loss.mean()
                    val_loss_value += single_loss
                    print_str += f"{name} - {single_loss}, "
            print(print_str[:-2])

        # Test the planner on a few random scenarios
        for test_idx in n_test:
            out_fn = f"{vis_out_dir}/{epoch:09d}_{test_idx:02d}.png"
            _, _ = test(planner, map, dataset, out_fn)
    return planner


if __name__ == "__main__":

    lr = 3e-4
    batch_size = 128
    n_support_points = 64
    num_train_steps = 500000
    n_test = 1
    vis_out_dir = "vis/diffusion"

    # If not exit, create output directory
    if not os.path.isdir(vis_out_dir):
        os.system(f"mkdir -p {vis_out_dir}")
        # This is a linux command which cannot be used on Windows
        # For me, I may use it to instead:
        # os.makedirs(vis_out_dir, exist_ok=True)
        # But it doesn't matter now
        # (ಡωಡ)hiahiahia

    # Choose GPU / CPU
    # tbh, linux NVIDIA driver is really difficult to install
    # So NVIDIA, fk U (Linus Voice)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load map and trajectory dataset
    map = ImageMap2D("data/2d_maze_2.png")
    dataset = TrajectoryDataset(data_dir="data/traj_data/BiRRT")

    # Create and initialize the planner model and optimizer
    unet = TemporalUnet(
        n_support_points=n_support_points,
        state_dim=2,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
    ).to(device)
    planner = GaussianDiffusion(
        model=unet,
        variance_schedule="exponential",
        n_diffusion_steps=25,
        predict_epsilon=True,
    ).to(device)
    criterion = GaussianDiffusionLoss().loss_fn
    # The torch.optim.AdamW optimizer is similar to Adam but incorporates weight decay directly
    # AdamW = Adam + weight decate
    # Emmm, Adam / AdamW, which is better?
    optimizer = torch.optim.Adam(lr=lr, params=planner.parameters())

    # Check if the 'diffusion.pt' file exists, otherwise train the model
    ckpt_fn = "data/diffusion.pt"
    if not os.path.isfile(ckpt_fn):
        planner = train(
            planner,
            dataset,
            criterion,
            optimizer,
            batch_size,
            num_train_steps,
            device,
            n_test,
            map,
            vis_out_dir,
        )
        torch.save(planner, ckpt_fn)
    else:
        # Load model and test it
        planner.model.load_state_dict(
            torch.load(ckpt_fn).model.to(device).state_dict()
        )
        print(
            f"==> Loaded pretrained checkpoint from {ckpt_fn}. Start testing..."
        )
        for test_idx in range(50):
            out_fn = f"{vis_out_dir}/final_{test_idx:02d}.png"
            paths, valid_paths = test(planner, map, dataset, out_fn=None)
            input(
                f"{test_idx}: start position [{paths[0][0][0]}, {paths[0][0][1]}], end position [{paths[0][-1][0]}, {paths[0][-1][1]}], {sum(valid_paths)} / {len(valid_paths)} valid path planned.\nPress enter to continue."
            )
