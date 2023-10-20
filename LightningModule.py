from torch import optim, nn
import torch
import torch.utils.data as data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.models as models
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


# The LightningModule
class LitDigitReader(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        for param in self.model.parameters():
            param.requires_grad = False
        num_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(num_filters, 10)
        for param in self.model.fc.parameters():
            param.requires_grad = True
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=1e-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(
        #     x.size(0), -1
        # )  # Get rid of unneeded dimensions in the image
        output = self(x)
        loss = self.loss_fn(output, y)  # Calculate the difference
        self.log("train_loss", loss)  # Log to TensorBoard
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(
        #     x.size(0), -1
        # )  # Get rid of unneeded dimensions in the image
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("val_loss", loss)  # Log to TensorBoard
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(
        #     x.size(0), -1
        # )  # Get rid of unneeded dimensions in the image
        output = self(x)
        loss = self.loss_fn(output, y)  # Calculate the difference
        self.log("test_loss", loss)  # Log to TensorBoard
        return loss


# Transforms
def make_3_channels(x):
    return x.repeat(3, 1, 1)


def main():
    # The actual model
    model = LitDigitReader()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(make_3_channels)]
    )

    # The training and validation data
    train_set = MNIST(
        root="MNIST", download=True, train=True, transform=transform
    )
    train_set_size = int(
        len(train_set) * 0.8
    )  # Setup train/valid dataset sizes
    valid_set_size = len(train_set) - train_set_size
    seed = torch.Generator().manual_seed(42)  # Setup RNG for train/valid split
    train_set, valid_set = data.random_split(  # Split train/valid datasets
        train_set, [train_set_size, valid_set_size], generator=seed
    )
    test_set = MNIST(
        root="MNIST", download=True, train=False, transform=transform
    )
    train_loader = data.DataLoader(train_set, batch_size=64, num_workers=10)
    valid_loader = data.DataLoader(valid_set, batch_size=64, num_workers=10)
    test_loader = data.DataLoader(test_set, batch_size=64, num_workers=10)

    # Train the model
    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=5,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    trainer.fit(model, train_loader, valid_loader)

    # Test the model
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
