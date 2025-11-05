from UpscalingImages import Upscaling
from Data import UpscaleDataset
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
import tqdm
import torch

# Used strictly for development to see the images
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train(epochs=1000, lr=0.001, save_every=100, loss_every=10, batch_size=32):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  dataset = UpscaleDataset(samples=10)
  model, epoch = Upscaling.load("model.pt")
  model = model.to(device)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  loss = nn.MSELoss()
  adam = optim.Adam(model.parameters(), lr=lr)
  total_loss = 0
  n_losses = 0
  batch = 0

  for i in range(epoch, epochs):
    for j, (batch_input, batch_target) in enumerate(loader):
      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)
      adam.zero_grad()
      output = model(batch_input)
      loss_val = loss(output, batch_target)
      total_loss += loss_val.item()
      n_losses += 1
      loss_val.backward()
      adam.step()

      batch += 1

      if batch % loss_every == 0:
        print(f"Loss: {total_loss / n_losses}")
        total_loss = 0
        n_losses = 0
        # figs, axs = plt.subplots(1,2)
        # axs[0].axis("off")
        # axs[1].axis("off")
        # axs[0].imshow(output[0].cpu().detach().permute(1, 2, 0))
        # axs[1].imshow(batch_target[0].cpu().detach().permute(1, 2, 0))
        # plt.show()

      if batch % save_every == 0:
        print(f"Saving model at epoch {i+1} on batch {j}/{len(loader)}")
        Upscaling.save(model, "model.pt", i)
        
train()