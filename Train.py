import math

from UpscalingImages import Upscaling
from Data import UpscaleDataset
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Used strictly for development to see the images
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train(epochs=10000, lr=0.0001, save_every=1, batch_size=32):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  # Loads model from filepath or creates new model
  model, epoch = Upscaling.load("Models/compress_model.pt")
  model = model.to(device)

  # Create Dataset and send into dataloader
  dataset = UpscaleDataset(color=False)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  # Initiate Loss and Optimizer functions
  loss_fn = nn.MSELoss()
  adam = optim.Adam(model.parameters(), lr=lr)
  total_loss = 0.0
  n_losses = 0

  # Loop through epochs, starting at 0 or where model left off
  for i in range(epoch, epochs):
    progress_bar = tqdm(loader, desc=f"Epoch {i}/{epochs}")
    running_loss = 0.0

    for j, (batch_input, batch_target) in enumerate(progress_bar):
      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)

      # Find values
      adam.zero_grad()
      output = model(batch_input)

      # (-1, 1) to (0, 1)
      output = (output + 1) / 2

      # Compute Loss
      loss = loss_fn(output, batch_target)
      running_loss += loss.item()
      total_loss += loss.item()
      n_losses += 1

      # Update weights
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

      adam.step()

      # Display Loss and PSNR
      if n_losses == 20:
        loss = total_loss / n_losses
        PSNR = 10 * math.log10(1 / loss)
        progress_bar.set_postfix({"Avg Loss:": loss, "PSNR:": PSNR})
        total_loss = 0
        n_losses = 0

    # Save model every __ epochs
    if (i + 1) % save_every == 0:
      print(f"Saving model at epoch {i}")
      Upscaling.save(model, "Models/compress_model.pt", i)

    if 0.01 > (running_loss / len(loader)):
      compression = max(50, dataset.get_compression() - 10)
      if compression > 50: print(f'Lowering Compression: {compression}')
      dataset.set_compression(compression)


def validate(test_loader='Datasets/Manga/Test', samples = 10000, compress = 100):
  dataset = UpscaleDataset(samples=samples, color=False, filepath = test_loader)
  # dataset = UpscaleDataset(samples=32*10, color=False)
  dataset.set_compression(compress)
  model_a, _ = Upscaling.load("Models/compress_model.pt")
  model_b, _ = Upscaling.load("Models/compress_model.pt")

  loader = DataLoader(dataset, batch_size=1, shuffle=True)
  for batch_input, batch_target in loader:
    with torch.no_grad():
      output_a = model_a(batch_input)
      output_b = model_b(batch_input)

      # (-1, 1) to (0, 1)
      output_a = (output_a + 1) / 2
      output_b = (output_b + 1) / 2

      # Clip outputs
      output_a = torch.clamp(output_a, 0, 1)
      output_b = torch.clamp(output_b, 0, 1)

      figs, axs = plt.subplots(2, 3)
      for i in range(2):
        axs[i][0].set_title("Transformer")
        axs[i][1].set_title("Original")
        axs[i][2].set_title("Input")
        axs[i][1].imshow(batch_target[0].detach().permute(1, 2, 0), cmap="gray")
        axs[i][2].imshow(batch_input[0].detach().permute(1, 2, 0), cmap="gray")
        
        for j in range(3):
          axs[i][j].axis("off")

      axs[0][0].imshow(output_a[0].detach().permute(1, 2, 0), cmap="gray")
      axs[1][0].imshow(output_b[0].detach().permute(1, 2, 0), cmap="gray")

      plt.show()

# train()
validate()