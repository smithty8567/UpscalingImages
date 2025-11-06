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
  def __init__(self, embedding_dim=2048, feedforward_dim=2048 * 4, num_layers=1, num_heads=4, patch_size=8):
    super().__init__()

    self.params = [embedding_dim, feedforward_dim, num_layers, num_heads, patch_size]
    self.patch_size = patch_size

    self.initialConv = nn.Sequential(
      nn.Conv2d(1, 64, 5, 1, 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 32, 3, 1, 1),
      nn.BatchNorm2d(num_features=32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, 3, 1, 1),
      nn.ReLU(inplace=True),
    )

    self.patch_norm = nn.LayerNorm(32 * patch_size * patch_size)
    self.patch_embed = nn.Linear(32 * patch_size * patch_size, embedding_dim)
    self.patch_unembed = nn.Linear(embedding_dim, 32 * patch_size * patch_size)

    self.transformer1 = ST.TransformerEncoder(
      embedding_dim=embedding_dim,
      feedforward_dim=feedforward_dim,
      num_layers=num_layers,
      num_heads=num_heads
    )

    self.finalConvLayer = nn.Sequential(
      nn.Conv2d(32, 4, 1, 1, 0),
      nn.PixelShuffle(2),
      nn.BatchNorm2d(num_features=1),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):

    x = self.initialConv(x)
    x = data.to_patches(x, self.patch_size)
    
    # x = self.patch_norm(x)
    # x = self.patch_embed(x)
    x = self.transformer1(x)
    # x = self.patch_unembed(x)

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
      checkpoint = torch.load(path, weights_only=True)
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
