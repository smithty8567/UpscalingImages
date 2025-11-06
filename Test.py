from CNN import CNNModel
from UpscalingImages import Upscaling
from Data import UpscaleDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
  vit_model, _ = Upscaling.load("model.pt")
  cnn_model, _ = CNNModel.load("cnn_model.pt")
  vit_model = vit_model.to(device)
  cnn_model = cnn_model.to(device)
  loader = DataLoader(dataset, batch_size=100, shuffle=True)
  input, target = loader.__iter__().__next__()

  with torch.no_grad():
    for i in range(input.shape[0]):
      y, cbcr = get_model_input(input[i])
      y = y.to(device)
      output_y_vit = vit_model(y)
      output_y_cnn = cnn_model(y)
      output_ycbcr_vit = np.concatenate((output_y_vit.cpu().permute(0, 2, 3, 1)[0].detach().numpy(), cbcr), axis=2)
      output_ycbcr_cnn = np.concatenate((output_y_cnn.cpu().permute(0, 2, 3, 1)[0].detach().numpy(), cbcr), axis=2)
      output_image_vit = ycbcr_to_rgb(output_ycbcr_vit)
      output_image_cnn = ycbcr_to_rgb(output_ycbcr_cnn)
      output_image_vit = output_image_vit.clip(0, 1)
      output_image_cnn = output_image_cnn.clip(0, 1)
      
      # Showing current model output compared to target image
      target_image = target[i].permute(1, 2, 0).detach().numpy()
      input_image = input[i].permute(1, 2, 0).detach().numpy()
      downscaled_image = cv2.resize(input_image, (input_image.shape[0] * 2, input_image.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
      fig, axs = plt.subplots(1, 4, sharey=True)
      axs[0].imshow(downscaled_image)
      axs[1].imshow(target_image)
      axs[2].imshow(output_image_vit)
      axs[3].imshow(output_image_cnn)
      axs[0].set_title('Downscaled')
      axs[1].set_title('Original')
      axs[2].set_title('Upscaled (VIT)')
      axs[3].set_title('Upscaled (CNN)')
      for i in range(4):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
      plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0, wspace=0)
      plt.show()

test()