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


class GAN(nn.Module):
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
          model = GAN()
          model.load_state_dict(data['state_dict'])
          print(f"Loaded from checkpoint at epoch {data['epoch'] + 1}")
          return model, data['epoch']
      except Exception as e:
          print(f"Error loading checkpoint: {e}")
          print("Creating new model...")
          return GAN(), 0
