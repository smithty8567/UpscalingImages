from UpscalingImages import Upscaling
from Data import UpscaleDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
import torch

# Used strictly for development to see the images
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = UpscaleDataset(samples=10)
  model, epoch = Upscaling.load("model.pt")
  model = model.to(device)
  loader = DataLoader(dataset, batch_size=4, shuffle=True)
  
  for batch_input, batch_target in tqdm.tqdm(loader):
    with torch.no_grad():
      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)
      output = model(batch_input)
      
      # Showing current model output compared to target image
      target_image = batch_target.cpu().permute(0, 2, 3, 1)[0].detach().numpy()
      output_image = output.cpu().permute(0, 2, 3, 1)[0].detach().numpy()
      fig, axs = plt.subplots(1, 2)
      axs[0].imshow(output_image, cmap="gray")
      axs[1].imshow(target_image, cmap="gray")
      plt.show()

test()