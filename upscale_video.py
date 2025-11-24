import cv2
import torch
import torch
from esrgan import Generator
import configparser as cp
import sys
import numpy as np

def get_model(device="cpu"):
  config = cp.ConfigParser()
  config.read("config.ini")
  gen_filepath = config['MODEL']['generator']

  model = Generator.load(gen_filepath)[0]
  model = model.to(device)
  model.eval()

  return model

def live(size, device, flip=True, path = 0):
  model = get_model(device)

  cap = cv2.VideoCapture(path)
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 4)
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 4)
  fps = cap.get(cv2.CAP_PROP_FPS)
  writer = cv2.VideoWriter('output.mp4', -1, fps, (h, w * 2))

  if not cap.isOpened():
    print("Could not open webcam")
    exit()

  print("Press 'q' to quit.")

  while True:
    ret, frame = cap.read()
    if not ret: break
    frame = frame.astype(np.float32) / 255
    if flip: frame = frame[:,::-1]
    inp = frame
    inp_img = inp
    # hight, width, _  = frame.shape
    # if hight < width:
    #   diff = (width - hight)//2
    #   frame = frame[:,diff:width-diff]
    # if hight > width:
    #   diff = (hight - width)//2
    #   frame = frame[diff:hight-diff, :]
    inp = cv2.resize(
      frame,
      (size, size),
      interpolation=cv2.INTER_NEAREST
    )
    # inp_img = cv2.resize(
    #   inp,
    #   (h, w),
    #   interpolation=cv2.INTER_NEAREST
    # )
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = torch.tensor(inp, dtype=torch.float32).permute(2,0,1)
    inp = inp.unsqueeze(0).to(device)  # [1,3,H,W]
    with torch.no_grad():
      out = model(inp)  # [1,3,H,W]
    out_img = out.squeeze().cpu().clamp(0, 1).permute(1,2,0)
    out_img = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
    pixelated = cv2.resize(
      out_img,
      (h, w),
      interpolation=cv2.INTER_NEAREST
    )
    combined = np.hstack((inp_img, pixelated))
    writer.write(combined)
    cv2.imshow("Camera (left)  |  Model Output (right)", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
  
  cap.release()
  writer.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  file = sys.argv[1] if len(sys.argv) > 1 else 0
  live(64, device, file == 0, file)