from torch import optim, nn
import torch
import torch.utils.data as data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import lightning.pytorch as pl


# The LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(
            x.size(0), -1
        )  # Get rid of unneeded dimensions in the image
        z = self.encoder(x)  # Encode the image
        x_hat = self.decoder(z)  # Decode the encoded image
        loss = nn.functional.mse_loss(x_hat, x)  # Calculate the difference
        self.log("train_loss", loss)  # Log to TensorBoard
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(
            x.size(0), -1
        )  # Get rid of unneeded dimensions in the image
        z = self.encoder(x)  # Encode the image
        x_hat = self.decoder(z)  # Decode the encoded image
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)  # Log to TensorBoard

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(
            x.size(0), -1
        )  # Get rid of unneeded dimensions in the image
        z = self.encoder(x)  # Encode the image
        x_hat = self.decoder(z)  # Decode the encoded image
        loss = nn.functional.mse_loss(x_hat, x)  # Calculate the difference
        self.log("test_loss", loss)  # Log to TensorBoard

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# The actual model
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
autoencoder = LitAutoEncoder(encoder, decoder)

# Transforms
transform = transforms.ToTensor()

# The training and validation data
train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
train_set_size = int(len(train_set) * 0.8)  # Setup train/valid dataset sizes
valid_set_size = len(train_set) - train_set_size
seed = torch.Generator().manual_seed(42)  # Setup RNG for train/valid split
train_set, valid_set = data.random_split(  # Split train/valid datasets
    train_set, [train_set_size, valid_set_size], generator=seed
)
test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
train_loader = data.DataLoader(train_set)
valid_loader = data.DataLoader(valid_set)
test_loader = data.DataLoader(test_set)

# Train the model
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(autoencoder, train_loader, valid_loader)

# Test the model
trainer.test(autoencoder, dataloaders=test_loader)
