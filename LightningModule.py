from torch import optim, nn
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

# The data
train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
train_loader = data.DataLoader(train_set)
test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
test_loader = data.DataLoader(test_set)

# Train the model
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# Test the model
trainer.test(autoencoder, dataloaders=test_loader)
