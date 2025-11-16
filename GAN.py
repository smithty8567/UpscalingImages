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
import torchvision

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
  def load(path, sr_path=None):
      try:
        data = torch.load(path, weights_only=True, map_location='cpu')
        model = Generator()
        model.load_state_dict(data['state_dict'])
        print(f"Loaded from checkpoint at epoch {data['epoch'] + 1} and iter {data['iter']}")
        return model, data['epoch'], data['iter']
      except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating new model...")
        model = Generator()
        if sr_path is not None:
          model.load_srresnet(sr_path)
        return model, 0, 0

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

    # Adaptive average pooling is not needed for input size 128x128
    # self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))

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
      model = Discriminator()
      model.load_state_dict(data['state_dict'])
      print(f"Loaded from checkpoint at epoch {data['epoch'] + 1} and iter {data['iter']}")
      return model, data['epoch'], data['iter']
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      print("Creating new model...")
      return Discriminator(), 0, 0

class PerceptualLoss(nn.Module):
  def __init__(self):
    super().__init__()
    
    # Stop at the 4th convolution before the 5th maxpool
    i = 5 # Max pool
    j = 4 # Conv

    # Load the pre-trained VGG19 available in torchvision
    vgg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    features = list(vgg19.features.children())

    # Keep track of what maxpool and convolution layer we are at
    maxpool_counter = 0
    conv_counter = 0
    truncate_at = 0
    
    # Iterate through the convolutional section ("features") of the VGG19
    for layer in features:
      truncate_at += 1

      # Count the number of maxpool layers and the convolutional layers after each maxpool
      if isinstance(layer, nn.Conv2d):
        conv_counter += 1
      if isinstance(layer, nn.MaxPool2d):
        maxpool_counter += 1
        conv_counter = 0

      # Break if we reach the jth convolution after the (i - 1)th maxpool
      if maxpool_counter == i - 1 and conv_counter == j: break

    # Check if conditions were satisfied
    assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (i, j)

    # Truncate to the jth convolution (skip activation) before the ith maxpool layer
    self.truncated_vgg19 = nn.Sequential(*features[:truncate_at])

  def forward(self, x, y):
    if x.shape[1] == 1:
      # Repeat the grayscale image 3 times (B, C, H, W)
      x = x.repeat(1, 3, 1, 1)
      y = y.repeat(1, 3, 1, 1)

    x = self.truncated_vgg19(x)
    y = self.truncated_vgg19(y)

    return torch.mean((x - y) ** 2)

class L1Loss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, y):
    return torch.mean(torch.abs(x - y))

def interpolate_models(model_a: nn.Module, model_b: nn.Module, alpha=0.5):
  model_c = type(model_a)()
  state_a = model_a.state_dict()
  state_b = model_b.state_dict()
  state_c = model_c.state_dict()

  for key in state_a.keys():
    state_c[key] = state_a[key] * (1 - alpha) + state_b[key] * alpha

  model_c.load_state_dict(state_c)
  return model_c

def set_lr(opt, iter):
  lr = 0.0001
  milestones = [50_000, 100_000, 200_000, 300_000]
  for milestone in milestones:
    if iter >= milestone:
      lr *= 0.5
  for param_group in opt.param_groups:
    param_group['lr'] = lr

def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # Data
  dataset = UpscaleDataset(filepath="Datasets/Manga/Train", in_size=64, out_size=128, color=False)
  loader = DataLoader(dataset, batch_size=16, shuffle=True)
  
  # Models
  gen, epoch, iter = Generator.load("Models/sr_gen_3.pt", "Models/sr_model.pt")
  dis = Discriminator.load("Models/sr_dis_3.pt")[0]
  gen = gen.to(device)
  dis = dis.to(device)
  
  # Optimizers (lr is scheduled so it doesn't matter)
  gen_opt = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.9, 0.999))
  dis_opt = optim.Adam(dis.parameters(), lr=0.0001, betas=(0.9, 0.999))
  
  # Losses
  l1_loss_fn = L1Loss()
  perceptual_loss_fn = PerceptualLoss()
  perceptual_loss_fn.to(device)
  
  # Training parameters
  gen_total_loss = 0
  dis_total_loss = 0
  adv_total_loss = 0
  prc_total_loss = 0
  l1_total_loss = 0
  
  for i in range(epoch, 10000):
    prog_bar = tqdm(loader)
    for j, (batch_input, batch_target) in enumerate(prog_bar):
      iter += 1
      
      set_lr(gen_opt, iter)
      set_lr(dis_opt, iter)

      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)
      
      if (iter % 3) != 0:
        # 1) Train Discriminator
        dis_opt.zero_grad()
        sr = gen(batch_input).detach()
        
        real_logits = dis(batch_target)
        fake_logits = dis(sr)

        real_mean = real_logits.mean()
        fake_mean = fake_logits.mean()

        # Softplus is numerically stable
        # log(sigmoid(x)) = -softplus(-x)
        # Relativistic adversarial loss
        dis_loss = (
          F.softplus(-(real_logits - fake_mean)).mean() +
          F.softplus(fake_logits - real_mean).mean()
        )

        dis_loss.backward()
        torch.nn.utils.clip_grad_norm_(dis.parameters(), 1.0)
        dis_opt.step()

        dis_total_loss += dis_loss.item()
      else:
        # 2) Train Generator
        gen_opt.zero_grad()
        sr = gen(batch_input)

        real_logits = dis(batch_target)
        fake_logits = dis(sr)

        real_mean = real_logits.mean()
        fake_mean = fake_logits.mean()

        # Relativistic adversarial loss
        adv_loss = (
          F.softplus(-(fake_logits - real_mean)).mean() +
          F.softplus(real_logits - fake_mean).mean()
        )

        l1_loss = l1_loss_fn(sr, batch_target)
        perceptual_loss = perceptual_loss_fn(sr, batch_target)
        gen_loss = perceptual_loss + 0.005 * l1_loss + 0.001 * adv_loss

        gen_loss.backward()
        gen_opt.step()

        gen_total_loss += gen_loss.item()
        adv_total_loss += adv_loss.item()
        prc_total_loss += perceptual_loss.item()
        l1_total_loss += l1_loss.item()

      if iter % 100 == 0:
        gen_loss, gen_total_loss = gen_total_loss / 100, 0
        dis_loss, dis_total_loss = dis_total_loss / 100, 0
        adv_loss, adv_total_loss = adv_total_loss / 100, 0
        prc_loss, prc_total_loss = prc_total_loss / 100, 0
        l1_loss, l1_total_loss = l1_total_loss / 100, 0
        prog_bar.set_postfix(gen_loss=gen_loss, dis_loss=dis_loss, adv_loss=adv_loss, prc_loss=prc_loss, l1_loss=l1_loss)
        Generator.save(gen, "Models/sr_gen_3.pt", i, iter)
        Discriminator.save(dis, "Models/sr_dis_3.pt", i, iter)

def test():
  model_a = Generator.load("Models/sr_gen_3.pt")[0]
  model_b = Generator.load("", "Models/sr_model.pt")[0]
  # model_c = interpolate_models(model_a, model_b, 0.3)
  test_model(model_a, model_b, 64, 128)

# train()
test()
