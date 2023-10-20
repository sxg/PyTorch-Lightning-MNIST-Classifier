import torch
from LightningModule import LitAutoEncoder, encoder, decoder

# Load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(
    checkpoint, encoder=encoder, decoder=decoder
)

# Choose the trained model
model = autoencoder.encoder
model.eval()

# Embed fake images
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = model(fake_image_batch)
print("Predictions for fake images:\n", embeddings)
