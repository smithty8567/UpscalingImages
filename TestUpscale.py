from torch.utils.data import DataLoader
from UpscalingImages import Upscaling
from Data import UpscaleDataset
import torch
import matplotlib.pyplot as plt
import os
import cv2

def test_model(model, old_model=None, in_size=64, out_size=128, color=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  if old_model is not None: old_model = old_model.to(device)
  dataset = UpscaleDataset(in_size=in_size, out_size=out_size, color=color)
  loader = DataLoader(dataset, batch_size=1, shuffle=True)

  for batch_input, batch_target in loader:
    batch_input = batch_input.to(device)
    batch_target = batch_target.to(device)

    with torch.no_grad():
      output = model(batch_input)
      if old_model is not None: old_output = old_model(batch_input)

    output = torch.clamp(output, 0, 1)
    if old_model is not None: old_output = torch.clamp(old_output, 0, 1)

    output_cpu = output.detach().cpu().permute(0, 2, 3, 1).numpy()
    if old_model is not None: old_output_cpu = old_output.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_cpu = batch_target.detach().cpu().permute(0, 2, 3, 1).numpy()
    input_cpu = batch_input.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    output_image = output_cpu[0].clip(0, 1)
    if old_model is not None: old_output_image = old_output_cpu[0].clip(0, 1)
    target_image = target_cpu[0].clip(0, 1)
    input_image = input_cpu[0].clip(0, 1)
    
    fig, axs = plt.subplots(1, 4 if old_model is not None else 3, figsize=(15, 5))

    def save_output(event):
      if event.key == 'w':
        n_files = len(os.listdir('Images'))
        cv2.imwrite(f"Images/output_{n_files}.png", (output_image * 255).astype('uint8'))

    fig.canvas.mpl_connect('key_press_event', save_output)

    axs[0].imshow(input_image, cmap="gray", interpolation='nearest')
    axs[1].imshow(target_image, cmap="gray")
    axs[2].imshow(output_image, cmap="gray")
    if old_model is not None: axs[3].imshow(old_output_image, cmap="gray")
    axs[0].set_title("Input")
    axs[1].set_title("Target")
    axs[2].set_title("SRGAN")
    if old_model is not None: axs[3].set_title("SRResNet")
    for i in range(4 if old_model is not None else 3):
      axs[i].set_xticks([])
      axs[i].set_yticks([])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0, wspace=0)
    plt.waitforbuttonpress()
    plt.show()
