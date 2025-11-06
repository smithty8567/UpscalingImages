from UpscalingImages import Upscaling
from Data import UpscaleDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
import torch
import cv2
import numpy as np

# Used strictly for development to see the images
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def rgb_to_ycbcr(image: np.ndarray):
  return cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

def ycbcr_to_rgb(image: np.ndarray):
  return cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)

def get_model_input(input: torch.Tensor):
  ycbcr = rgb_to_ycbcr(input.cpu().permute(1, 2, 0).detach().numpy())
  y = ycbcr[..., 0]
  cbcr = ycbcr[..., 1:]
  input = torch.from_numpy(y).unsqueeze(2).permute(2, 0, 1).unsqueeze(0)
  return input, cv2.resize(cbcr, (cbcr.shape[0] * 2, cbcr.shape[1] * 2))

def test():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = UpscaleDataset('Datasets/Cartoon/Test', color=True)
  model, epoch = Upscaling.load("model.pt")
  model = model.to(device)
  loader = DataLoader(dataset, batch_size=100, shuffle=True)
  input, target = loader.__iter__().__next__()

  with torch.no_grad():
    for i in range(input.shape[0]):
      y, cbcr = get_model_input(input[i])
      y = y.to(device)
      output_y = model(y)
      output_ycbcr = np.concatenate((output_y.cpu().permute(0, 2, 3, 1)[0].detach().numpy(), cbcr), axis=2)
      output_image = ycbcr_to_rgb(output_ycbcr)
      
      # Showing current model output compared to target image
      target_image = target[i].permute(1, 2, 0).detach().numpy()
      input_image = input[i].permute(1, 2, 0).detach().numpy()
      bicubic_image = cv2.resize(input_image, (input_image.shape[0] * 2, input_image.shape[1] * 2))
      fig, axs = plt.subplots(1, 3)
      axs[0].imshow((target_image + 1) / 2)
      axs[1].imshow((output_image + 1) / 2)
      axs[2].imshow((bicubic_image + 1) / 2)
      axs[0].set_title('Original')
      axs[1].set_title('Upscaled')
      axs[2].set_title('Bicubic')
      plt.show()

test()