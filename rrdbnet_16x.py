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
from rrdbnet_4x import RRDBNet, UpscaleBlock

class RRDBNet16x(nn.Module):
  """
  RRDBNet variant for 16x upscaling.
  This module extends the RRDBNet by adding an additional UpscaleBlock.
  """
  def __init__(self, num_blocks=23, channels=64, growth=32):
    super().__init__()
    self.rrdbnet = RRDBNet(num_blocks, channels, growth)
    self.upscale = UpscaleBlock()

    self.upscale = nn.Sequential(
      UpscaleBlock(64, scale=2),
      UpscaleBlock(64, scale=2)
    )

    self.final = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    # Initialize final layer with zeros
    self.final.weight.data.zero_()
    self.final.bias.data.zero_()

  def forward(self, x):
    x = self.rrdbnet.forward_features(x)
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
  def load(path, rrdbnet_path=None):
    try:
      model = RRDBNet16x()
      data = torch.load(path, weights_only=True, map_location='cpu')
      model.load_state_dict(data['state_dict'])
      print(f"Loaded from checkpoint at epoch {data['epoch'] + 1}")
      return model, data['epoch'], data['iter']
    except Exception as e:
      if rrdbnet_path is None: raise 'Error loading checkpoint and no RRDBNet path provided'
      model = RRDBNet16x()
      rrdbnet = RRDBNet.load(rrdbnet_path)[0]
      model.rrdbnet = rrdbnet
      print(f"Error loading checkpoint: {e}")
      print("Creating new model...")
      return model, 0, 0

def train(device):
  config = cp.ConfigParser()
  config.read("config.ini")
  rrdb4x_filepath = config['MODEL']['psnr_4x']
  rrdb16x_filepath = config['MODEL']['psnr_16x']
  data_filepath = config['DATA']['train_data']

  # Models
  model, epoch, iter = RRDBNet16x.load(rrdb16x_filepath, rrdb4x_filepath)
  model = model.to(device)

  # Data
  dataset = UpscaleDataset(filepath=data_filepath, in_size=64, out_size=256)
  loader = DataLoader(dataset, batch_size=16, shuffle=True)
  
  # Loss
  loss_fn = nn.L1Loss()
  scaler = torch.amp.GradScaler('cuda')
  adam = optim.Adam(model.parameters(), lr=0.00005)
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
        RRDBNet16x.save(model, rrdb16x_filepath, i, iter)

def test(device):
  config = cp.ConfigParser()
  config.read("config.ini")
  model = RRDBNet16x.load(config['MODEL']['psnr_16x'])[0]
  test_model(model, None, 64, 256, device)

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if "--test" in sys.argv:
    test(device)
  else:
    train(device)
