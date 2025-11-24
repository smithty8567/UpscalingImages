from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from data import UpscaleDataset
from test_model import test_model
from rrdbnet_16x import RRDBNet16x
from metrics_logger import MetricLogger
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SN
import torchvision
import lpips
import configparser as cp
import sys

# Initialize the generator with a pretrained RRDB network
class Generator(nn.Module):
  def __init__(self, num_blocks=23):
    super().__init__()

    self.sr = RRDBNet16x(num_blocks=num_blocks)

  def load_net(self, path):
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
          model.load_net(sr_path)
        return model, 0, 0

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Sequential(
      SN(nn.Conv2d(3, 64, 3, 1, 1)), # 256x256
      nn.LeakyReLU(0.2, inplace=True)
    )

    self.conv_block = nn.Sequential(
      SN(nn.Conv2d(64, 64, 3, 2, 1)), # 128x128
      nn.LeakyReLU(0.2, inplace=True),

      SN(nn.Conv2d(64, 128, 3, 1, 1)),
      nn.LeakyReLU(0.2, inplace=True),

      SN(nn.Conv2d(128, 128, 3, 2, 1)), # 64x64
      nn.LeakyReLU(0.2, inplace=True),

      SN(nn.Conv2d(128, 256, 3, 1, 1)),
      nn.LeakyReLU(0.2, inplace=True),

      SN(nn.Conv2d(256, 256, 3, 2, 1)), # 32x32
      nn.LeakyReLU(0.2, inplace=True),

      SN(nn.Conv2d(256, 512, 3, 1, 1)),
      nn.LeakyReLU(0.2, inplace=True),

      SN(nn.Conv2d(512, 512, 3, 2, 1)), # 16x16
      nn.LeakyReLU(0.2, inplace=True),

      SN(nn.Conv2d(512, 512, 3, 1, 1)),
      nn.LeakyReLU(0.2, inplace=True),

      SN(nn.Conv2d(512, 512, 3, 2, 1)), # 8x8
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

def set_lr(opt, iter):
  lr = 0.0001
  milestones = [50_000, 100_000, 200_000, 300_000]
  for milestone in milestones:
    if iter >= milestone:
      lr *= 0.5
  for param_group in opt.param_groups:
    param_group['lr'] = lr

def train(device):
  config = cp.ConfigParser()
  config.read("config.ini")
  train_filepath = config['DATA']['train_data']
  psnr_16x_filepath = config['MODEL']['psnr_16x']
  gen_filepath = config['MODEL']['generator']
  dis_filepath = config['MODEL']['discriminator']

  # Data
  dataset = UpscaleDataset(filepath=train_filepath, in_size=64, out_size=256)
  loader = DataLoader(dataset, batch_size=10, shuffle=True)
  
  # Models
  gen, epoch, iter = Generator.load(gen_filepath, psnr_16x_filepath)
  dis = Discriminator.load(dis_filepath)[0]
  gen = gen.to(device)
  dis = dis.to(device)
  
  # Optimizers (lr is scheduled so it doesn't matter)
  gen_opt = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.9, 0.999))
  dis_opt = optim.Adam(dis.parameters(), lr=0.0001, betas=(0.9, 0.999))
  
  # Losses
  l1_loss_fn = nn.L1Loss()
  perceptual_loss_fn = lpips.LPIPS(net='vgg')
  if device.type == 'cuda': perceptual_loss_fn.cuda()
  
  # Training parameters
  log = MetricLogger('Logs/esrgan.csv', ['gen_loss', 'dis_loss', 'adv_loss', 'prc_loss', 'l1_loss'])
  gen_total_loss = 0
  dis_total_loss = 0
  adv_total_loss = 0
  prc_total_loss = 0
  l1_total_loss = 0
  n_loss_gen = 0
  n_loss_dis = 0

  for i in range(epoch, 10000):
    prog_bar = tqdm(loader)
    for j, (batch_input, batch_target) in enumerate(prog_bar):
      iter += 1
      
      set_lr(gen_opt, iter // 2)
      set_lr(dis_opt, iter // 2)

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

        log.log_metric('dis_loss', dis_loss.item())
        dis_total_loss += dis_loss.item()
        n_loss_dis += 1
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
        perceptual_loss = perceptual_loss_fn(sr * 2 - 1, batch_target * 2 - 1).mean()
        gen_loss = perceptual_loss + 0.05 * l1_loss + 0.001 * adv_loss

        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        gen_opt.step()

        log.log_metric('gen_loss', gen_loss.item())
        log.log_metric('adv_loss', adv_loss.item())
        log.log_metric('prc_loss', perceptual_loss.item())
        log.log_metric('l1_loss', l1_loss.item())
        gen_total_loss += gen_loss.item()
        adv_total_loss += adv_loss.item()
        prc_total_loss += perceptual_loss.item()
        l1_total_loss += l1_loss.item()
        n_loss_gen += 1

      if iter % 100 == 0:
        dis_loss, dis_total_loss = dis_total_loss / n_loss_dis, 0
        gen_loss, gen_total_loss = gen_total_loss / n_loss_gen, 0
        adv_loss, adv_total_loss = adv_total_loss / n_loss_gen, 0
        prc_loss, prc_total_loss = prc_total_loss / n_loss_gen, 0
        l1_loss, l1_total_loss = l1_total_loss / n_loss_gen, 0
        n_loss_dis = 0
        n_loss_gen = 0
        prog_bar.set_postfix(gen_loss=gen_loss, dis_loss=dis_loss, adv_loss=adv_loss, prc_loss=prc_loss, l1_loss=l1_loss)
        Generator.save(gen, gen_filepath, i, iter)
        Discriminator.save(dis, dis_filepath, i, iter)

      log.next_iter()

def test(device):
  config = cp.ConfigParser()
  config.read("config.ini")
  model_a = Generator.load(config['MODEL']['generator'])[0]
  test_model(model_a, None, 64, 256, device)

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if "--test" in sys.argv:
    test(device)
  else:
    train(device)
