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

def train(epochs=40, lr=0.001, save_every=1):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = UpscaleDataset(samples=10)
  model, epoch = Upscaling.load("model.pt")
  model = model.to(device)
  loader = DataLoader(dataset, batch_size=4, shuffle=True)
  loss = nn.MSELoss()
  adam = optim.Adam(model.parameters(), lr=lr)
  total_loss = 0
  n_losses = 0

  for i in range(epoch, epochs):
    print(f"Epoch {i+1}")
    for batch_input, batch_target in tqdm.tqdm(loader):
      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)
      adam.zero_grad()
      output = model(batch_input)
      loss_val = loss(output, batch_target)
      total_loss += loss_val.item()
      n_losses += 1
      loss_val.backward()
      adam.step()
    if (i+1) % save_every == 0:
      # Showing current model output compared to target image
      # target_image = batch_target.cpu().permute(0, 2, 3, 1)[0].detach().numpy()
      # output_image = output.cpu().permute(0, 2, 3, 1)[0].detach().numpy()
      # fig, axs = plt.subplots(1, 2)
      # axs[0].imshow(output_image, cmap="gray")
      # axs[1].imshow(target_image, cmap="gray")
      # plt.show()

      print(f"Saving model at epoch {i+1} with loss {total_loss / n_losses}")
      Upscaling.save(model, "model.pt", i)
      total_loss = 0
      n_losses = 0

train()