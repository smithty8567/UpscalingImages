import os
import cv2
import torch
import torch
from esrgan import Generator
import configparser as cp
import sys
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip

def get_model(device="cpu"):
  config = cp.ConfigParser()
  config.read("config.ini")
  gen_filepath = config['MODEL']['generator']

  model = Generator.load(gen_filepath)[0]
  model = model.to(device)
  model.eval()

  return model

def make_video(infile=0, outfile="output.mp4", downsize=1, device="cpu"):
  reader = cv2.VideoCapture(infile)
  w = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) / downsize) * 4
  h = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) / downsize) * 4
  min_width = 1000
  if infile != 0:
    fps = reader.get(cv2.CAP_PROP_FPS)
    frame_size = (w * 2, h)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(outfile, codec, fps, frame_size)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    prog_bar = tqdm(range(n_frames))
  model = get_model(device)
  while True:
    if infile != 0: prog_bar.update(1)
    ret, frame = reader.read()
    if not ret: break
    inp = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
    inp_img = cv2.resize(inp, (w, h), interpolation=cv2.INTER_CUBIC)
    with torch.no_grad():
      inp_t = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
      inp_t = torch.tensor(inp_t, dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255
      out = model(inp_t.to(device))
      out_img = out.squeeze().cpu().permute(1,2,0).numpy()
      out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
      out_img = (out_img * 255).clip(0, 255).astype(np.uint8)
    combined = np.hstack((inp_img, out_img))
    if combined.shape[1] < min_width:
      new_width = min_width
      new_height = int(new_width / combined.shape[1] * combined.shape[0])
      combined = cv2.resize(combined, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    if infile != 0: writer.write(combined)
    cv2.imshow("Input (left)  |  Upscaled (right)", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
  cv2.destroyAllWindows()
  reader.release()
  if infile != 0:
    writer.release()
    add_audio_to_video(infile, outfile)

def add_audio_to_video(input_video, output_video):
  try:
    outfile = os.path.splitext(output_video)[0]
    audio = AudioFileClip(input_video)
    video = VideoFileClip(output_video)
    video = video.set_audio(audio)
    video.write_videofile(f"{outfile}_with_audio.mp4", codec='libx264', audio_codec='aac', verbose=False, logger=None)
    os.remove(output_video)
  except Exception as e:
    print(f"Error adding audio to video: {e}")

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  downsize = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  infile = sys.argv[2] if len(sys.argv) > 2 else 0
  outfile = sys.argv[3] if len(sys.argv) > 3 else "output.mp4"
  make_video(infile, outfile, downsize, device)
