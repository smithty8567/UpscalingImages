import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from UpscalingImages import Upscaling
from CNN import CNNModel
from Data import UpscaleDataset
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
import tqdm
import torch


def train(epochs=300, lr=0.0001, save_every=2, loss_every=2, batch_size=32):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  dataset = UpscaleDataset(samples=90000)
  model, epoch = Upscaling.load("model.pt")
  model = model.to(device)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  loss = nn.MSELoss()
  adam = optim.Adam(model.parameters(), lr=lr)
  total_loss = 0
  n_losses = 0
  batch = 0
  for i in range(epoch, epochs):
    print(f"Epoch {i + 1}")
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

      batch += 1

    if (i+1) % loss_every == 0:
      print(f"Loss: {total_loss / n_losses}")
      total_loss = 0
      n_losses = 0
      # figs, axs = plt.subplots(1,2)
      # axs[0].axis("off")
      # axs[1].axis("off")
      # axs[0].imshow(output[0].cpu().detach().permute(1, 2, 0))
      # axs[1].imshow(batch_target[0].cpu().detach().permute(1, 2, 0))
      # plt.show()

    if (i+1) % save_every == 0:
      # print(f"Saving model at epoch {i+1} on batch {batch}/{len(loader)}")
      print("Model is saving...")
      Upscaling.save(model, "model.pt", i)

def validate(test_loader= 'Datasets/Cartoon/Test', samples = 5000):
    dataset = UpscaleDataset(samples=samples, filepath = test_loader)
    model, epoch = Upscaling.load("model1.pt")
    cnn_model, epoch2 = CNNModel.load("cnn_model.pt")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_input, batch_target in loader:
        with torch.no_grad():
            output = model(batch_input)
            output2 = cnn_model(batch_input)
            figs, axs = plt.subplots(1, 3,figsize=(15, 4))
            axs[0].axis("off")
            axs[1].axis("off")
            axs[2].axis("off")

            axs[0].set_title("Transformer")
            axs[1].set_title("Original")
            axs[2].set_title("Bicubic")

            axs[0].imshow(output[0].cpu().detach().squeeze(), cmap="gray")
            axs[1].imshow(batch_target[0].cpu().detach().squeeze(), cmap="gray")
            axs[2].imshow(output2[0].cpu().detach().squeeze(), cmap="gray")
            plt.show()


# train()
validate()