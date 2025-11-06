import torch
import torch.nn as nn
import torchinfo
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

import Transformer as ST
import Data as data

# ===============================
# Transformer-based Upscaling Model
# ===============================
class Upscaling(nn.Module):
  def __init__(self, embedding_dim=2048, num_layers=4, num_heads=4, patch_size=8, channel_size = 8):
    super().__init__()

    channels = [channel_size, channel_size * 2, channel_size * 4]
    self.params = [embedding_dim, num_layers, num_heads, patch_size,channel_size]
    self.patch_size = patch_size
    self.initialConv = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=channels[0]),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=channels[2]),
        nn.ReLU(inplace=True)
    )

    self.transformer1 = ST.TransformerEncoder(
      embedding_dim=embedding_dim,
      feedforward_dim=embedding_dim*4,
      num_layers=num_layers,
      num_heads=num_heads
    )
    self.finalConvLayer = nn.Sequential(
      nn.Conv2d(in_channels=channels[2], out_channels=channels[1], kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=channels[1]),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=channels[1], out_channels=1, kernel_size=3, padding=1),
      nn.ConvTranspose2d(1,1,kernel_size=3,stride=2, padding=1, output_padding=1),
      nn.BatchNorm2d(num_features=1),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):

    # Adding more channels to the image
    # print("BEFORE CONV", x.shape)
    x = self.initialConv(x)

    # shape the image into patches
    # print("BEFORE PATCHES", x.shape)
    x = data.to_patches(x, self.patch_size)
    # print("Shape before transformer:", x.shape)
    x = self.transformer1(x)
    # print("Shape after transformer:", x.shape)

    x = data.to_image(x, self.patch_size)
    x = self.finalConvLayer(x)
    return x

  @staticmethod
  def save(model, path, epoch):
    try:
      torch.save({
        'state_dict': model.state_dict(),
        'params': model.params,
        'epoch': epoch
      }, path + '.tmp')
      os.replace(path + '.tmp', path)
    except Exception as e:
      print(f"Error saving checkpoint: {e}")

  @staticmethod
  def load(path):
    try:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      checkpoint = torch.load(path, map_location=device,weights_only=True)
      params = checkpoint['params']
      model = Upscaling(*params)
      model.load_state_dict(checkpoint['state_dict'])
      print(f"Loaded from checkpoint at epoch {checkpoint['epoch'] + 1}")
      return model, checkpoint['epoch']
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      print("Creating new model...")
      return Upscaling(), 0

# torchinfo.summary(Upscaling(), input_size=(32,1,64,64))
