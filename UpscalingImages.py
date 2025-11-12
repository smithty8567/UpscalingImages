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
  def __init__(self, num_layers=4, num_heads=2, patch_size=8, multiply_channel = 8):
    super().__init__()

    # Embedding dimension is patch_size * patch_size * num_channels
    # num_channels = channel_size * 4

    init_channels = 3 # Color
    channels = [multiply_channel, multiply_channel * 2, multiply_channel * 4, multiply_channel * 8]
    self.params = [num_layers, num_heads, patch_size,multiply_channel]
    self.patch_size = patch_size
    self.initialConv = nn.Sequential(
        nn.Conv2d(in_channels=init_channels, out_channels=channels[0], kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=channels[0]),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=channels[2]),
        nn.ReLU(inplace=True)
    )

    embedding_dim = patch_size * patch_size * channels[2]
    self.transformer1 = ST.TransformerEncoder(
      embedding_dim=embedding_dim,
      feedforward_dim=embedding_dim * 2,
      num_layers=num_layers,
      num_heads=num_heads
    )
    self.residualConv = nn.Sequential(
        nn.Conv2d(in_channels=channels[2], out_channels=channels[2], kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channels[2], out_channels=channels[2], kernel_size=3, padding=1),
    )
    self.convLayer2 = nn.Sequential(
      nn.Conv2d(in_channels=channels[2], out_channels=channels[1], kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(channels[1], channels[1], kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.BatchNorm2d(num_features=channels[1]),
      nn.Conv2d(in_channels=channels[1], out_channels=channels[1], kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
    )

    embedding_dim = patch_size * patch_size * channels[1]
    self.transformer2 = ST.TransformerEncoder(
      embedding_dim=embedding_dim,
      feedforward_dim=embedding_dim * 2 ,
      num_layers=num_layers // num_layers,
      num_heads=num_heads
    )

    self.convLayer3 = nn.Sequential(
      nn.Conv2d(in_channels=channels[1], out_channels=channels[1], kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=channels[1]),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=channels[1], out_channels=channels[0], kernel_size=3, padding=1),
      nn.ConvTranspose2d(channels[0],channels[0],kernel_size=3,stride=2, padding=1, output_padding=1),
      nn.BatchNorm2d(num_features=channels[0]),
      nn.ReLU(inplace=True),
    )

    embedding_dim = patch_size * patch_size * channels[0]
    self.transformer3 = ST.TransformerEncoder(
        embedding_dim=embedding_dim,
        feedforward_dim=embedding_dim * 2,
        num_layers=num_layers // num_layers,
        num_heads=num_heads
    )

    self.finalConvLayer = nn.Sequential(
      nn.Conv2d(in_channels=channels[0], out_channels=3, kernel_size=3, padding=1),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):

    # Adding more channels to the image
    x = self.initialConv(x) # (32, 3, 64, 64) --> (32, 32, 64, 64)
    # print("After CONV", x.shape)

    x = data.to_patches(x, self.patch_size) # (32, 32, 64, 64) --> (32, 64, 2048)
    # print("After PATCHES", x.shape)
    x = self.transformer1(x) # (32, 64, 2048)
    # print("Shape after transformer:", x.shape)
    x = data.to_image(x, self.patch_size) # (32, 64, 2048) --> (32, 32, 64, 64)
    # print("Back into image:", x.shape)
    res = self.residualConv(x)
    x = x + res
    x = self.convLayer2(x) # (32, 32, 64, 64) --> (32, 16, 128, 128)
    # print("After 2nd conv:", x.shape)


    x = data.to_patches(x, self.patch_size) # (32, 16, 128, 128) --> (32, 256, 1024)
    # print("After patches again:", x.shape)
    x = self.transformer2(x) # (32, 256, 1024)
    # print("After 2nd transformer:", x.shape)
    x = data.to_image(x, self.patch_size) # (32, 256, 1024) --> (32, 16, 128, 128)
    # print("Image again:", x.shape)

    x = self.convLayer3(x)

    x = data.to_patches(x, self.patch_size)  # (32, 16, 128, 128) --> (32, 1024, 512)
    # print("After patches again:", x.shape)
    x = self.transformer3(x)  # (32, 1024, 512)
    # print("After 3rd transformer:", x.shape)
    x = data.to_image(x, self.patch_size)  # (32, 1024, 512) --> (32, 8, 256, 256)
    # print("Image again:", x.shape)

    x = self.finalConvLayer(x) # (32, 8, 256, 256) --> (32, 3, 256, 256)
    # print("final image shape:", x.shape)
    x = torch.sigmoid(x) # Renormalizes values between 0 and 1

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
