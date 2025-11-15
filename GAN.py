from torch.utils.data import DataLoader
from torch import optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from Data import UpscaleDataset
from TestUpscale import test_model
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import SRResNet as SR

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
    x = self.sr(x)
    return x

  @staticmethod
  def load(path, sr_path=None):
      try:
        data = torch.load(path, weights_only=True, map_location='cpu')
        model = Generator()
        model.load_state_dict(data['state_dict'])
        print(f"Loaded from checkpoint at epoch {data['epoch'] + 1}")
        return model, data['epoch']
      except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating new model...")
        model = Generator()
        if sr_path is not None:
          model.load_srresnet(sr_path)
        return model, 0

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(1, 64, 9, 1, 4), # 128x128
      nn.LeakyReLU(0.2, inplace=True)
    )

    self.conv_block = nn.Sequential(
      nn.Conv2d(64, 64, 3, 2, 1), # 64x64
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(64, 128, 3, 1, 1),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(128, 128, 3, 2, 1), # 32x32
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(128, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(256, 256, 3, 2, 1), # 16x16
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(256, 512, 3, 1, 1),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(512, 512, 3, 2, 1), # 8x8
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2, inplace=True)
    )

    self.linear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512 * 8 * 8, 1024),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(1024, 1)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv_block(x)
    x = self.linear(x)
    return x

  @staticmethod
  def save(model, path, epoch):
    torch.save({
      'epoch': epoch,
      'state_dict': model.state_dict()
    }, path + '.tmp')
    os.replace(path + '.tmp', path)

  @staticmethod
  def load(path):
    try:
      data = torch.load(path, weights_only=True, map_location='cpu')
      model = Discriminator()
      model.load_state_dict(data['state_dict'])
      print(f"Loaded from checkpoint at epoch {data['epoch'] + 1}")
      return model, data['epoch']
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      print("Creating new model...")
      return Discriminator(), 0

def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = UpscaleDataset(filepath="Datasets/Manga/Train")
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  gen, epoch = Generator.load("Models/sr_gen.pt", "Models/sr_model.pt")
  dis, _ = Discriminator.load("Models/sr_dis.pt")
  gen = gen.to(device)
  dis = dis.to(device)
  gen_opt = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.9, 0.999))
  dis_opt = optim.Adam(dis.parameters(), lr=0.0001, betas=(0.9, 0.999))
  total_loss = 0
  n_losses = 0
  batch = 0

  # Losses
  adversarial_loss = nn.BCEWithLogitsLoss()
  upscale_loss = SR.CharbonnierLoss()

  for i in range(epoch, 10000):
    prog_bar = tqdm(loader)
    for j, (batch_input, batch_target) in enumerate(prog_bar):
      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)
      
      if j % 2 == 0:
        # 1) Train Discriminator
        dis_opt.zero_grad()
        sr = gen(batch_input).detach()
        
        # Fake loss
        fake_prob = dis(sr)
        fake_loss = adversarial_loss(fake_prob, torch.zeros_like(fake_prob))

        # Real loss
        real_prob = dis(batch_target)
        real_loss = adversarial_loss(real_prob, torch.ones_like(real_prob) * 0.9)

        # Total loss
        dis_loss = (fake_loss + real_loss) * 0.5
        dis_loss.backward()
        dis_opt.step()
      else:
        # 2) Train Generator
        gen_opt.zero_grad()
        sr = gen(batch_input)
        prob = dis(sr)

        # Combine adversarial and upscale losses
        adv_loss = adversarial_loss(prob, torch.ones_like(prob))
        rec_loss = upscale_loss(sr, batch_target)
        gen_loss = 0.001 * adv_loss + rec_loss

        gen_loss.backward()
        gen_opt.step()

        total_loss += gen_loss.item()
        n_losses += 1
        batch += 1

      if batch % 100 == 0:
        print(f"Saving model at epoch {i+1} on batch {j}/{len(prog_bar)} with loss {total_loss/n_losses}")
        Generator.save(gen, "Models/sr_gen.pt", i)
        Discriminator.save(dis, "Models/sr_dis.pt", i)
        total_loss = 0
        n_losses = 0

def test():
  model = Generator.load("Models/sr_gen.pt")
  test_model(model, 64, 128)

train()
# test()
