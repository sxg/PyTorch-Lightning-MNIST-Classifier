from torch import optim, nn
import torch
import lightning.pytorch as pl
from torchmetrics import functional as F
from torchvision.utils import make_grid
from DigitDataModule import DigitDataModule


# The LightningModule
class DigitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_logged_images = False
        self.valid_logged_images = False
        self.test_logged_images = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_reshaped = x.view(-1, 784)
        output = self(x_reshaped)
        loss = self.loss_fn(output, y)  # Calculate the difference
        self.log("train/loss", loss)  # Log to TensorBoard

        # Log images to Tensorboard
        if not self.train_logged_images:
            preds = torch.argmax(output, dim=1)
            img_grid = make_grid(x)
            self.logger.experiment.add_image(
                "train/inputs", img_grid, self.current_epoch
            )
            self.logger.experiment.add_text(
                "train/targets", str(y), self.current_epoch
            )
            self.logger.experiment.add_text(
                "train/preds", str(preds), self.current_epoch
            )
            self.train_logged_images = True

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_reshaped = x.view(-1, 784)
        output = self(x_reshaped)
        loss = self.loss_fn(output, y)
        self.log("val/loss", loss)  # Log to TensorBoard
        acc = F.accuracy(output, y, task="multiclass", num_classes=10)
        self.log("val/acc", acc)  # Log to Tensorboard

        # Log images to Tensorboard
        if not self.valid_logged_images:
            preds = torch.argmax(output, dim=1)
            img_grid = make_grid(x)
            self.logger.experiment.add_image(
                "val/inputs", img_grid, self.current_epoch
            )
            self.logger.experiment.add_text(
                "val/targets", str(y), self.current_epoch
            )
            self.logger.experiment.add_text(
                "val/preds", str(preds), self.current_epoch
            )
            self.valid_logged_images = True

        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_reshaped = x.view(-1, 784)
        output = self(x_reshaped)
        loss = self.loss_fn(output, y)  # Calculate the difference
        self.log("test/loss", loss)  # Log to TensorBoard
        acc = F.accuracy(output, y, task="multiclass", num_classes=10)
        self.log("test/acc", acc)  # Log to Tensorboard

        # Log images to Tensorboard
        if not self.test_logged_images:
            preds = torch.argmax(output, dim=1)
            img_grid = make_grid(x)
            self.logger.experiment.add_image(
                "test/inputs", img_grid, self.current_epoch
            )
            self.logger.experiment.add_text(
                "test/targets", str(y), self.current_epoch
            )
            self.logger.experiment.add_text(
                "test/preds", str(preds), self.current_epoch
            )
            self.test_logged_images = True

        return {"loss": loss, "acc": acc}

    def on_training_epoch_end(self):
        self.train_logged_images = False

    def on_validation_epoch_end(self):
        self.valid_logged_images = False

    def on_test_epoch_end(self):
        self.test_logged_images = False


def main():
    # The actual model
    model = DigitModule()

    # The data
    dm = DigitDataModule(batch_size=64)

    # Train the model
    trainer = pl.Trainer(
        max_epochs=5,
    )
    trainer.fit(model, datamodule=dm)

    # Test the model
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
