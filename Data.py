import os
import torch
import tqdm as tqdm
import random
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import encode_jpeg, decode_jpeg

def generate_directory_list(filepath, samples):
  """Creates array of image directories."""
  directories = []
  for root, dirs, files in tqdm.tqdm(os.walk(filepath)):
    for file in files:
      path = os.path.join(root, file)
      path = path.replace("/", "\\")
      directories.append(path)
      if samples is not None and len(directories) > samples:
        break
  return directories

class ImageProcessing:
  def __init__(self):
    pass

  def get_target_image(self, image, out_size):
    """
    Input : HWC, BGR, 0-255, numpy
    Output : CHW, RGB, 0-1, tensor
    """
    
    # Inter nearest excluded to avoid misalignment issues
    interpolation = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])

    # Find a starting size that is at least out_size
    start_size = random.randint(out_size, min(image.shape[0], image.shape[1]))

    # Random start_size x start_size image
    x = random.randint(0, image.shape[0] - start_size)
    y = random.randint(0, image.shape[1] - start_size)
    image = image[x:x+start_size, y:y+start_size]

    # Resize to out_size
    image = cv2.resize(image, (out_size, out_size), interpolation=interpolation)

    # Convert to tensor
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.clip(0, 1)

    return image
  
  def get_input_image(self, in_size, image):
    """
    Input : CHW, RGB, 0-1, tensor
    Output : CHW, RGB, 0-1, tensor
    """

    # Convert to numpy
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Inter nearest excluded to avoid misalignment issues
    interpolation = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])

    # Apply blur
    if random.random() < 0.2:
      kernel = random.choice([3,5])
      sigma_x = random.uniform(0.1, 3.0)
      sigma_y = random.uniform(0.1, 3.0)
      image = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma_x, sigmaY=sigma_y)

    # Resize to in_size
    image = cv2.resize(image, (in_size, in_size), interpolation=interpolation)

    # Add noise (gaussian, poisson)
    if random.random() < 0.2:
      noise_type = random.choice(['color', 'bw'])
      image = image / 255.0
      dims = 3 if noise_type == 'color' else 1
      mean = 0
      var = random.uniform(0.00005, 0.002)
      sigma = var ** 0.5
      gauss = np.random.normal(mean, sigma, (image.shape[0], image.shape[1], dims))
      image = image + gauss
      image = np.clip(image, 0, 1)
      image = (image * 255).astype(np.uint8)

    # Convert to tensor
    image = torch.from_numpy(image).permute(2, 0, 1)

    # JPEG Compression
    if random.random() < 0.5:
      quality = random.randint(70, 100)
      jpeg_image = encode_jpeg(image, quality=quality)
      image = decode_jpeg(jpeg_image).float()
    
    # Normalize and clip
    image = image / 255.0
    image = image.clip(0, 1)
    
    return image

class UpscaleDataset(Dataset):
  """
    Dataset class for the Upscaling Image Transformer.

    Args:
      filepath (string): Path to images.
      in_size (int): Length of the input sequence.
      out_size (int): Length of the output sequence.
      color (boolean): Dimension of embedding vectors.
      samples (int): Number of Images

    Returns:
      __len__ (int): Length of the dataset.
      __getitem__ (int): processed image and original image from index.
    """

  def __init__(self, filepath, in_size = 64, out_size = 128, color = False, samples=None):
    self.directories = generate_directory_list(filepath, samples)
    self.in_size = in_size
    self.out_size = out_size
    self.color = color
    self.image_processing = ImageProcessing()
    
  def __len__(self):
    return len(self.directories)

  def __getitem__(self, idx):
    image = self.directories[idx]
    if not self.color: raise NotImplementedError("Grayscale images not implemented in UpscaleDataset.")
    cv_image = cv2.imread(image)
    target_image = self.image_processing.get_target_image(cv_image, self.out_size)
    input_image = self.image_processing.get_input_image(self.in_size, target_image)
    return input_image, target_image

def segment_images():
  from_path = 'Datasets/Wallpapers/Train2'
  to_path = 'Datasets/Wallpapers/Train3'
  dirs = generate_directory_list(from_path, None)

  if not os.path.exists(to_path):
    os.makedirs(to_path)

  for path in tqdm.tqdm(dirs):
    file_name = path.split('\\')[-1]
    image = cv2.imread(path)
    
    # Cut in half
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    
    # Divide into four pieces
    middle_x = image.shape[1] // 2
    middle_y = image.shape[0] // 2
    tr = image[:middle_y, :middle_x]
    tl = image[:middle_y, middle_x:]
    br = image[middle_y:, :middle_x]
    bl = image[middle_y:, middle_x:]

    # Save images
    cv2.imwrite(f'{to_path}/tr_{file_name}', tr)
    cv2.imwrite(f'{to_path}/tl_{file_name}', tl)
    cv2.imwrite(f'{to_path}/br_{file_name}', br)
    cv2.imwrite(f'{to_path}/bl_{file_name}', bl)

def filter_images():
  # Remove all images with height/width < 256
  path = 'Datasets/Wallpapers/Test3'
  dirs = generate_directory_list(path, None)

  for path in tqdm.tqdm(dirs):
    image = cv2.imread(path)
    if image.shape[0] < 256 or image.shape[1] < 256:
      os.remove(path)

# segment_images()
# filter_images()