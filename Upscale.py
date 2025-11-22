import cv2
import numpy as np
import torch
from ESRGAN import Generator, interpolate_models

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_a = Generator.load("Models/sr_gen_wallpapers_8.pt")[0]
# model_b = Generator.load("", "Models/sr_rrdb.pt")[0]
# model_c = interpolate_models(model_a, model_b, 0.3)
model = model_a

model = model.to(device)
model.eval()

rgb = cv2.imread("input.jpg")
rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2))
rgb = cv2.imwrite("input_resized.png", rgb)
rgb = torch.from_numpy(rgb).unsqueeze(0).permute(0, 3, 1, 2).float() / 255
rgb = rgb.to(device)

output_image = model(rgb)

output_image = output_image.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
output_image = (output_image * 255).clip(0, 255).astype('uint8')

# rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2))
# y, crcb = get_model_input(rgb)
# y = y.to(device).detach()
# y = model(y)
# output_image = get_model_output(y, crcb)

cv2.imwrite("output.png", output_image)

