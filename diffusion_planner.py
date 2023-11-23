import collections

import torch
from torch.utils.data import DataLoader, random_split

from diffusion_model import GaussianDiffusion, GaussianDiffusionLoss, TemporalUnet
from traj_dataset import TrajectoryDataset


def dict_to_device(ob, device):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)


def train(
    model, dataset, criterion, optimizer, batch_size, num_train_steps, device
):

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    num_epochs = int(num_train_steps * batch_size / len(train_dataset))

    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            batch = dict_to_device(batch, device)
            train_loss_dict, train_loss_info = criterion(
                model, batch, train_dataset.dataset
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        for idx, batch in enumerate(val_dataloader):
            batch = dict_to_device(batch, device)
            with torch.no_grad():
                val_loss_dict, val_loss_info = criterion(
                    model, batch, val_dataset.dataset
                )
                val_loss_value = 0.0
                print_str = f"Eval batch [{idx}]: "
                for name, loss in val_loss_dict.items():
                    single_loss = loss.mean()
                    val_loss_value += single_loss
                    print_str += f"{name} - {single_loss}, "
            print(print_str[:-2])


if __name__ == "__main__":

    lr = 3e-4
    batch_size = 128
    num_train_steps = 500000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(data_dir="data/traj_data/BiRRT")
    unet = TemporalUnet(
        n_support_points=64,
        state_dim=2,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
    ).to(device)
    model = GaussianDiffusion(
        model=unet,
        variance_schedule="exponential",
        n_diffusion_steps=25,
        predict_epsilon=True,
    ).to(device)
    criterion = GaussianDiffusionLoss().loss_fn
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    train(
        model,
        dataset,
        criterion,
        optimizer,
        batch_size,
        num_train_steps,
        device,
    )

    import ipdb

    ipdb.set_trace()
