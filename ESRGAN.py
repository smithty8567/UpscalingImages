import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from Data import UpscaleDataset
from TestUpscale import test_model

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

class DenseBlock(nn.Module):
  def __init__(self, channels=64, growth=32):
    super().__init__()

    self.convs = nn.ModuleList([
      nn.Conv2d(channels + i * growth, growth, 3, 1, 1)
      for i in range(5)
    ])

    self.activation = nn.LeakyReLU(0.2, inplace=True)
    self.final_conv = nn.Conv2d(channels + 5 * growth, channels, 1, 1, 0)

  def forward(self, x):
    features = [x]
    for conv in self.convs:
      out = self.activation(conv(torch.cat(features, dim=1)))
      features.append(out)
    out = torch.cat(features, dim=1)
    out = self.final_conv(out) * 0.2
    return out + x

class RRDB(nn.Module):
  def __init__(self, channels=64, growth=32):
    super().__init__()
    self.DB1 = DenseBlock(channels, growth)
    self.DB2 = DenseBlock(channels, growth)
    self.DB3 = DenseBlock(channels, growth)

  def forward(self, x):
    out = self.DB1(x)
    out = self.DB2(out)
    out = self.DB3(out)
    return out * 0.2 + x

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

class RRDBNet(nn.Module):
  def __init__(self, num_blocks=23, channels=64, growth=32):
    super().__init__()

    self.head = nn.Sequential(
      nn.Conv2d(3, channels, kernel_size=9, padding=4),
      nn.PReLU()
    )

    self.res_blocks = nn.Sequential(
      *[RRDB(channels, growth) for _ in range(num_blocks)]
    )

    self.trunk_conv = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    )

    self.upscale = nn.Sequential(
      UpscaleBlock(channels, scale=2),
    )

    self.final = nn.Conv2d(channels, 3, kernel_size=9, padding=4)

    # Initialize final layer with zeros
    self.final.weight.data.zero_()
    self.final.bias.data.zero_()

  def forward(self, x):
    x_head = self.head(x)
    x_res = self.res_blocks(x_head)
    x_trunk = self.trunk_conv(x_res)
    x = x_head + x_trunk
    x = self.upscale(x)
    x = self.final(x)
    return x

  @staticmethod
  def save(model, path, epoch, iter):
    try:
      torch.save({
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'iter': iter
      }, path + '.tmp')
      os.replace(path + '.tmp', path)
    except Exception as e:
      print(f"Error saving checkpoint: {e}")

  @staticmethod
  def load(path):
    try:
      data = torch.load(path, weights_only=True, map_location='cpu')
      model = RRDBNet()
      model.load_state_dict(data['state_dict'])
      print(f"Loaded from checkpoint at epoch {data['epoch'] + 1}")
      return model, data['epoch'], data['iter']
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      print("Creating new model...")
      return RRDBNet(), 0, 0

def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = UpscaleDataset(filepath="Datasets/Wallpapers/Train3", in_size=64, out_size=128, color=True)
  model, epoch, iter = RRDBNet.load("Models/sr_rrdb_wallpapers.pt")
  model = model.to(device)
  loader = DataLoader(dataset, batch_size=16, shuffle=True)
  loss_fn = nn.L1Loss()
  scaler = torch.amp.GradScaler('cuda')
  adam = optim.Adam(model.parameters(), lr=0.00002)
  total_loss = 0
  n_losses = 0
  last_loss = "?"
  last_saved = "Never"

  for i in range(epoch, 10000):
    prog_bar = tqdm(loader)
    for j, (batch_input, batch_target) in enumerate(prog_bar):
      iter += 1
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

      if iter % 50 == 0:
        last_loss = total_loss / n_losses
        prog_bar.set_postfix(loss=last_loss, saved=last_saved)
        total_loss = 0
        n_losses = 0
      
      if iter % 100 == 0:
        last_saved = f"Epoch {i+1} batch {j}"
        prog_bar.set_postfix(loss=last_loss, saved=last_saved)
        RRDBNet.save(model, "Models/sr_rrdb_wallpapers.pt", i, iter)

def test():
  model = RRDBNet.load("Models/sr_rrdb_wallpapers.pt")[0]
  test_model(model, None, 128, 512, True)

# train()
# test()
