import SRResNet as SR
import cv2
import numpy as np
import torch

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

model = SR.SRResNet.load("Models/sr_model.pt")[0]
rgb = cv2.imread("test.png")
y, crcb = get_model_input(rgb)
y = model(y)
output_image = get_model_output(y, crcb)
cv2.imwrite("output.png", output_image)

