import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from data import UpscaleDataset
from test_model import test_model
import configparser as cp
import sys

class DenseBlock(nn.Module):
  def __init__(self, channels=64, growth=32, beta=0.2):
    super().__init__()

    self.beta = beta
    self.convs = nn.ModuleList([
      nn.Conv2d(channels + i * growth, growth, 3, 1, 1)
      for i in range(4)
    ])

    self.activation = nn.LeakyReLU(0.2, inplace=True)
    self.final_conv = nn.Conv2d(channels + 4 * growth, channels, 1, 1, 0)

  def forward(self, x):
    features = [x]
    for i, conv in enumerate(self.convs):
      out = self.activation(conv(torch.cat(features, dim=1)))
      if i % 2 == 0:
        skip_connection = out
      else:
        out = skip_connection + out * self.beta
      features.append(out)
    out = torch.cat(features, dim=1)
    out = self.final_conv(out)
    return out

class RRDB(nn.Module):
  """
      Residual in Residual Dense Block class for the ESRGAN.

      Args:
        channels (int): Starting Channels for Dense Blocks.
        growth (int): Growth of the Convolutions in Dense Blocks.
        beta (float): Negative slope for LeakyReLU.

  """
  def __init__(self, channels=64, growth=32, beta=0.2):
    super().__init__()
    self.beta = beta
    self.DB1 = DenseBlock(channels, growth, beta)
    self.DB2 = DenseBlock(channels, growth, beta)
    self.DB3 = DenseBlock(channels, growth, beta)
    self.noise_factor_1 = nn.Parameter(torch.tensor(0.0))
    self.noise_factor_2 = nn.Parameter(torch.tensor(0.0))
    self.noise_factor_3 = nn.Parameter(torch.tensor(0.0))

  def forward(self, x):
    out = x
    out = out + self.DB1(out) * self.beta + self.noise_factor_1 * torch.randn_like(out)
    out = out + self.DB2(out) * self.beta + self.noise_factor_2 * torch.randn_like(out)
    out = out + self.DB3(out) * self.beta + self.noise_factor_3 * torch.randn_like(out)
    return out

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
  
  def forward_features(self, x):
    x_head = self.head(x)
    x_res = self.res_blocks(x_head)
    x_trunk = self.trunk_conv(x_res)
    x = x_head + x_trunk
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

def train(device):
  config = cp.ConfigParser()
  config.read("config.ini")
  rrdb4x_filepath = f"Models/{config['MODEL']['psnr_4x']}"
  
  # Models
  model, epoch, iter = RRDBNet.load(rrdb4x_filepath)
  model = model.to(device)

  # Data
  dataset = UpscaleDataset(filepath="Datasets/Wallpapers/Train3", in_size=64, out_size=256)
  loader = DataLoader(dataset, batch_size=16, shuffle=True)
  
  # Loss
  loss_fn = nn.L1Loss()
  scaler = torch.amp.GradScaler('cuda')
  adam = optim.Adam(model.parameters(), lr=0.0002)
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
        RRDBNet.save(model, rrdb4x_filepath, i, iter)

def test(device):
  config = cp.ConfigParser()
  config.read("config.ini")
  model = RRDBNet.load(config['MODEL']['psnr_4x'])[0]
  test_model(model, None, 64, 128, device)

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if "--test" in sys.argv:
    test(device)
  else:
    train(device)
