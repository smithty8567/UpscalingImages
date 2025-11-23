import cv2
import torch
from esrgan import Generator
import configparser as cp
import sys

def upscale_image(input_path="input.png", output_path="output.png", resize=None, device="cpu"):
  config = cp.ConfigParser()
  config.read("config.ini")
  gen_filepath = config['MODEL']['generator']

  model = Generator.load(gen_filepath)[0]
  model = model.to(device)
  model.eval()

  x = cv2.imread(input_path)
  
  if resize is not None:
    x = cv2.resize(x, (x.shape[1] // resize, x.shape[0] // resize))
    cv2.imwrite("resized.png", x)
  
  x = torch.from_numpy(x).unsqueeze(0).permute(0, 3, 1, 2).float() / 255
  x = x.to(device)

  sr = model(x)
  sr = sr.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
  sr = (sr * 255).clip(0, 255).astype('uint8')

  cv2.imwrite(output_path, sr)

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input_path = sys.argv[1] if len(sys.argv) > 1 else "input.png"
  output_path = sys.argv[2] if len(sys.argv) > 2 else "output.png"
  resize = int(sys.argv[3]) if len(sys.argv) > 3 else None
  upscale_image(input_path, output_path, resize, device)
