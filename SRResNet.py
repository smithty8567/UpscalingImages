import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from Data import UpscaleDataset
import TestUpscale

class CharbonnierLoss(nn.Module):
  """A differentiable version of L1 loss (Mean Absolute Error)
  that behaves like L2 loss when targets are close to zero.
  This makes it more robust to outliers than L2 loss while
  preserving differentiability at zero. The larger epsilon is,
  the less gradients near zero will matter (smoother images)."""
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def forward(self, pred, target):
    return torch.mean(torch.sqrt((pred - target)**2 + self.eps**2))

class ResidualBlock(nn.Module):
  def __init__(self, channels=64):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=3, padding=1),
      nn.PReLU(),
      nn.Conv2d(channels, channels, kernel_size=3, padding=1),
    )

  def forward(self, x):
    return x + self.block(x)

class UpscaleBlock(nn.Module):
  def __init__(self, channels=64, scale=2):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(channels, channels * (scale ** 2), kernel_size=3, padding=1),
      nn.PixelShuffle(scale),
      nn.PReLU()
    )

  def forward(self, x):
    return self.block(x)

class SRResNet(nn.Module):
  def __init__(self, num_blocks=16):
    super().__init__()

    self.head = nn.Sequential(
      nn.Conv2d(1, 64, kernel_size=9, padding=4),
      nn.PReLU()
    )

    self.res_blocks = nn.Sequential(
      *[ResidualBlock(64) for _ in range(num_blocks)]
    )

    self.trunk_conv = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, padding=1)
    )

    self.upscale = UpscaleBlock(64, scale=2)

    self.final = nn.Conv2d(64, 1, kernel_size=9, padding=4)

  def forward(self, x):
    x_head = self.head(x)
    x_res = self.res_blocks(x_head)
    x_trunk = self.trunk_conv(x_res)
    x = x_head + x_trunk
    x = self.upscale(x)
    x = self.final(x)
    return x

  @staticmethod
  def save(model, path, epoch):
    try:
      torch.save({
        'state_dict': model.state_dict(),
        'epoch': epoch
      }, path + '.tmp')
      os.replace(path + '.tmp', path)
    except Exception as e:
      print(f"Error saving checkpoint: {e}")

  @staticmethod
  def load(path):
    try:
      data = torch.load(path, weights_only=True, map_location='cpu')
      model = SRResNet()
      model.load_state_dict(data['state_dict'])
      print(f"Loaded from checkpoint at epoch {data['epoch'] + 1}")
      return model, data['epoch']
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      print("Creating new model...")
      return SRResNet(), 0

def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = UpscaleDataset(filepath="Datasets/Manga/Train")
  model, epoch = SRResNet.load("Models/sr_model_2.pt")
  model = model.to(device)
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  loss_fn = CharbonnierLoss()
  scaler = torch.amp.GradScaler('cuda')
  adam = optim.Adam(model.parameters(), lr=0.00005)
  total_loss = 0
  n_losses = 0
  batch = 0
  last_loss = "?"
  last_saved = "Never"

  for i in range(epoch, 10000):
    prog_bar = tqdm(loader)
    for j, (batch_input, batch_target) in enumerate(prog_bar):
      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)

      adam.zero_grad()
      
      with torch.amp.autocast('cuda'):
        output = model(batch_input)
        loss_val = loss_fn(output, batch_target)
      
      scaler.scale(loss_val).backward()
      scaler.step(adam)
      scaler.update()

      total_loss += loss_val.item()
      n_losses += 1
      batch += 1

      if batch % 500 == 0:
        last_loss = total_loss / n_losses
        prog_bar.set_postfix(loss=last_loss, saved=last_saved)
        total_loss = 0
        n_losses = 0
      
      if batch % 2000 == 0:
        last_saved = f"Epoch {i+1} batch {j}"
        prog_bar.set_postfix(loss=last_loss, saved=last_saved)
        SRResNet.save(model, "Models/sr_model_2.pt", i)

def test():
  model = SRResNet.load("Models/sr_model.pt")[0]
  TestUpscale.test_model(model, None, 64, 128)

# train()
test()