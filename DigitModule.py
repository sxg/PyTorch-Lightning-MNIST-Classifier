from torch import optim, nn
import torch
import torchvision.models as models
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics import functional as F
from torchvision.utils import make_grid
from DigitDataModule import DigitDataModule


# The LightningModule
class DigitModule(pl.LightningModule):
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
        self.valid_logged_images = False
        self.test_logged_images = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)  # Calculate the difference
        self.log("train_loss", loss)  # Log to TensorBoard
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("val_loss", loss)  # Log to TensorBoard
        acc = F.accuracy(output, y, task="multiclass", num_classes=10)
        self.log("val_acc", acc)  # Log to Tensorboard

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
        output = self(x)
        loss = self.loss_fn(output, y)  # Calculate the difference
        self.log("test_loss", loss)  # Log to TensorBoard
        acc = F.accuracy(output, y, task="multiclass", num_classes=10)
        self.log("test_acc", acc)  # Log to Tensorboard

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
        limit_train_batches=100,
        max_epochs=5,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    trainer.fit(model, datamodule=dm)

    # Test the model
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
