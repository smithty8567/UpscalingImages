import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from Data import UpscaleDataset
import SRResNet as SR

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 9, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 128 * 128, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_block(x)
        x = self.final_conv(x)
        x = self.linear(x)

        return x

class Generator(nn.Module):
  def __init__(self, num_blocks=16):
    super().__init__()

    self.sr = SR.SRResNet(num_blocks=num_blocks)

  def load_srresnet(self, path):
      try:
          data = torch.load(path, weights_only=True, map_location='cpu')
          self.sr.load_state_dict(data['state_dict'])
          print(f"Loaded from checkpoint at epoch {data['epoch'] + 1}")
      except Exception as e:
          print(f"Error loading checkpoint: {e}")

  def forward(self, x):
    x = self.sr(x) #low-res image to high-res image
    return x

  @staticmethod
  def load(path):
      try:
          data = torch.load(path, weights_only=True, map_location='cpu')
          model = Generator()
          model.load_state_dict(data['state_dict'])
          print(f"Loaded from checkpoint at epoch {data['epoch'] + 1}")
          return model, data['epoch']
      except Exception as e:
          print(f"Error loading checkpoint: {e}")
          print("Creating new model...")
          return Generator(), 0
