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

def get_model_output(y: torch.Tensor, cbcr: np.ndarray):
  ycbcr = np.concatenate((y.cpu().permute(0, 2, 3, 1)[0].detach().numpy(), cbcr), axis=2)
  output_image = ycbcr_to_rgb(ycbcr)
  output_image = output_image.clip(0, 1)
  return output_image

def test():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = UpscaleDataset('Datasets/Manga/Test', color=False)
  vit_model, _ = Upscaling.load("Models/manga_model_2.pt")
  cnn_model, _ = CNNModel.load("Models/cnn_model.pt")
  vit_model = vit_model.to(device)
  cnn_model = cnn_model.to(device)
  loader = DataLoader(dataset, batch_size=100, shuffle=True)
  input, target = loader.__iter__().__next__()

  with torch.no_grad():
    for i in range(input.shape[0]):
      # y, cbcr = get_model_input(input[i])
      y = input[i].unsqueeze(0)
      y = y.to(device)
      output_y_vit = vit_model(y)
      output_y_cnn = cnn_model(y)
      output_image_vit = output_y_vit.cpu().permute(0, 2, 3, 1)[0].clip(0, 1)
      output_image_cnn = output_y_cnn.cpu().permute(0, 2, 3, 1)[0].clip(0, 1)
      # output_image_vit = get_model_output(output_y_vit, cbcr)
      # output_image_cnn = get_model_output(output_y_cnn, cbcr)
      
      # Showing current model output compared to target image
      target_image = target[i].permute(1, 2, 0).detach().numpy()
      input_image = input[i].permute(1, 2, 0).detach().numpy()
      downscaled_image = cv2.resize(input_image, (input_image.shape[0] * 2, input_image.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
      fig, axs = plt.subplots(1, 4, sharey=True)
      axs[0].imshow(downscaled_image, cmap="gray")
      axs[1].imshow(target_image, cmap="gray")
      axs[2].imshow(output_image_vit, cmap="gray")
      axs[3].imshow(output_image_cnn, cmap="gray")
      axs[0].set_title('Downscaled')
      axs[1].set_title('Original')
      axs[2].set_title('Upscaled (VIT)')
      axs[3].set_title('Upscaled (CNN)')
      for i in range(4):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
      plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0, wspace=0)
      plt.show()

# test()