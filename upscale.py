import cv2
import torch
from esrgan import Generator
import configparser as cp
import sys

def get_model(device="cpu"):
  config = cp.ConfigParser()
  config.read("config.ini")
  gen_filepath = config['MODEL']['generator']

  model = Generator.load(gen_filepath)[0]
  model = model.to(device)
  model.eval()

  return model

def upscale_patches(img, model, w=256, h=256, margin=32, device="cpu"):
  inp = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # BCHW
  inp = inp.to(device)

  H, W = img.shape[:2]
  scale = 4
  out_H, out_W = H * scale, W * scale

  output = torch.zeros((1, 3, out_H, out_W), device="cpu")

  with torch.no_grad():
    for i in range(0, H, h - margin):
      for j in range(0, W, w - margin):

        # Clamp patch bounds
        i0 = max(i - margin, 0)
        j0 = max(j - margin, 0)
        i1 = min(i + h + margin, H)
        j1 = min(j + w + margin, W)

        patch = inp[:, :, i0:i1, j0:j1]  # BCHW

        # Run model
        sr = model(patch) # 4x upscaling
        sr = sr.cpu()

        # Crop scaled-by-4 margin
        top = (i - i0) * scale
        left = (j - j0) * scale
        bottom = top + h * scale
        right = left + w * scale

        sr_cropped = sr[:, :, top:bottom, left:right]

        # Place into final output
        out_i0 = i * scale
        out_j0 = j * scale
        out_i1 = out_i0 + sr_cropped.shape[2]
        out_j1 = out_j0 + sr_cropped.shape[3]

        # Clip to avoid overflow at borders
        out_i1 = min(out_i1, out_H)
        out_j1 = min(out_j1, out_W)

        output[:, :, out_i0:out_i1, out_j0:out_j1] = sr_cropped[:, :, :out_i1-out_i0, :out_j1-out_j0]

  # Convert to BGR uint8
  output = output.squeeze(0).permute(1, 2, 0).numpy()
  output = (output * 255).clip(0, 255).astype("uint8")
  return output

def upscale_image(img, model, device="cpu"):
  img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float() / 255
  img = img.to(device)

  with torch.no_grad():
    sr = model(img)
  
  sr = sr.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
  sr = (sr * 255).clip(0, 255).astype('uint8')

  return sr

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  input_path = sys.argv[1] if len(sys.argv) > 1 else "input.png"
  output_path = sys.argv[2] if len(sys.argv) > 2 else "output.png"
  resize = int(sys.argv[3]) if len(sys.argv) > 3 else None
  s = int(sys.argv[4]) if len(sys.argv) > 4 else 256
  margin = int(sys.argv[5]) if len(sys.argv) > 5 else 32
  model = get_model(device)
  img = cv2.imread(input_path)
  if resize is not None and resize != 1:
    img = cv2.resize(img, (img.shape[1] // resize, img.shape[0] // resize))
    cv2.imwrite("resized.png", img)
  out = upscale_patches(img, model, s, s, margin, device)
  # out = upscale_image(img, model, device)
  cv2.imwrite(output_path, out)
