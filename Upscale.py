import SRResNet as SR
import cv2
import numpy as np
import torch
from GAN import Generator, interpolate_models

def get_model_input(rgb: np.ndarray):
  ycrcb = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCR_CB)
  y = ycrcb[..., 0]
  crcb = ycrcb[..., 1:]
  y = torch.from_numpy(y).unsqueeze(2).permute(2, 0, 1).unsqueeze(0).float() / 255
  return y, crcb

def get_model_output(y: torch.Tensor, crcb: np.ndarray):
  y = y.cpu().permute(0, 2, 3, 1)[0].detach().numpy()
  y = (y * 255).clip(0, 255).astype('uint8')
  crcb = cv2.resize(crcb, (y.shape[1], y.shape[0]))
  output_image = y
  ycrcb = np.concatenate((y, crcb), axis=2)
  output_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
  output_image = output_image
  return output_image

# model = Generator.load("Models/sr_gen_3.pt")[0]
model_a = Generator.load("Models/sr_gen_3.pt")[0]
model_b = Generator.load("", "Models/sr_model.pt")[0]
model_c = interpolate_models(model_a, model_b, 0.3)
model = model_b

rgb = cv2.imread("test.jpg")
y, crcb = get_model_input(rgb)
y = model(y)
output_image = get_model_output(y, crcb)
cv2.imwrite("output3.png", output_image)

