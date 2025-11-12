from UpscalingImages import Upscaling
from Data import UpscaleDataset
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

class CNNModel(nn.Module):
  def __init__(self):
    super().__init__()
    # Inplace reduces memory usage
    # PixelShuffle converts (B, 4, H, W) to (B, 1, 2H, 2W)
    self.model = nn.Sequential(
      nn.Conv2d(1, 64, 5, 1, 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 32, 3, 1, 1),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, 3, 1, 1),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 4, 1, 1, 0),
      nn.PixelShuffle(2)
    )

  def forward(self, x):
    return F.tanh(self.model(x))
  
  @staticmethod
  def save(model, path, epoch):
    try:
      torch.save({
        'state_dict': model.state_dict(),
        'epoch': epoch
      }, path)
    except Exception as e:
      print(f"Error saving checkpoint: {e}")

  @staticmethod
  def load(path):
    try:
      data = torch.load(path, weights_only=True)
      model = CNNModel()
      model.load_state_dict(data['state_dict'])
      print(f"Loaded from checkpoint at epoch {data['epoch'] + 1}")
      return model, data['epoch']
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      print("Creating new model...")
      return CNNModel(), 0
  
def train(epochs=10000, lr=0.001, save_every=50, batch_size=16):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = UpscaleDataset()
  model, epoch = CNNModel.load("cnn_model.pt")
  model = model.to(device)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  loss = nn.MSELoss()
  adam = optim.Adam(model.parameters(), lr=lr)
  total_loss = 0
  n_losses = 0

  for i in range(epoch, epochs):
    print(f"Epoch {i+1}")
    batch = 0
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
      if batch % save_every == 0:
        print(f" Saving model at epoch {i+1} on batch {batch}/{len(dataset)/4} with loss {total_loss/n_losses}")
        CNNModel.save(model, "cnn_model.pt", i)
        total_loss = 0
        n_losses = 0

def test():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = UpscaleDataset()
  model, epoch = CNNModel.load("cnn_model.pt")
  model = model.to(device)
  loader = DataLoader(dataset, batch_size=4, shuffle=True)
  
  for batch_input, batch_target in tqdm.tqdm(loader):
    with torch.no_grad():
      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)
      output = model(batch_input)
      
      # Showing current model output compared to target image
      target_image = batch_target.cpu().permute(0, 2, 3, 1)[0].detach().numpy()
      output_image = output.cpu().permute(0, 2, 3, 1)[0].detach().numpy()
      fig, axs = plt.subplots(1, 2)
      axs[0].imshow((output_image + 1) / 2, cmap="gray")
      axs[1].imshow((target_image + 1) / 2, cmap="gray")
      plt.show()

# train()
# test()