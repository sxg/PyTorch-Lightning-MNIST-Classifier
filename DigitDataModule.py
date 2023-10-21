import lightning.pytorch as pl
import torch
import torch.utils.data as data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


class DigitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        # Setup the transforms
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(DigitDataModule._make_3_channels),
                # transforms.RandomRotation(15),
                transforms.RandomAffine(10, (0.05, 0.05), (0.8, 1.2), 5),
                transforms.RandomCrop(26),
                transforms.RandomErasing(0.5, (0.02, 0.1), (0.3, 3.3)),
                # transforms.Lambda(DigitDataModule._add_noise),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(DigitDataModule._make_3_channels),
            ]
        )

    # Helper function to replicate single channel images to 3 channel
    def _make_3_channels(x):
        return x.repeat(3, 1, 1)

    def _add_noise(x):
        return x + 0.1 * torch.randn_like(x)

    def prepare_data(self):
        MNIST("MNIST", train=True, download=True)
        MNIST("MNIST", train=False, download=True)

    def setup(self, stage):
        if stage == "fit":
            full_set = MNIST(
                root="MNIST",
                train=True,
                transform=self.train_transform,
            )
            train_set_size = int(len(full_set) * 0.8)
            val_set_size = len(full_set) - train_set_size
            seed = torch.Generator().manual_seed(42)
            (
                self.train_set,
                self.val_set,
            ) = data.random_split(  # Split train/val datasets
                full_set, [train_set_size, val_set_size], generator=seed
            )
        elif stage == "test":
            self.test_set = MNIST(
                root="MNIST",
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=10
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=10
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=10
        )
