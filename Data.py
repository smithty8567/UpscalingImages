import os
import torch
import tqdm as tqdm
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt

def to_patches(image_batch, patch_size):
    # Image Batch Shape: (Batch, Color Channels, Height, Width)
    # patched_batch Shape: (Batch, (height * width)/patch_size^2, color *  patch_size^2)
    batch = image_batch.shape[0]
    color = image_batch.shape[1]
    height, width = image_batch.shape[-2:]

    patched_batch = image_batch.view(batch, color, height // patch_size, patch_size, width // patch_size, patch_size)
    patched_batch = patched_batch.permute(0, 2, 4, 1, 3, 5)
    patched_batch = patched_batch.contiguous().view(batch, height // patch_size * width // patch_size,
                                                        color * patch_size ** 2)
    return patched_batch

def to_image(patched_batch, patch_size):
    # patched_batch Shape: (Batch, (height * width)/patch_size^2, color *  patch_size^2)
    # Image Batch Shape: (Batch, Color Channels, Height, Width)
    batch = patched_batch.shape[0]
    color = patched_batch.shape[2] // (patch_size * patch_size)
    height = patched_batch.shape[1]
    width = patched_batch.shape[1]

    image_batch = patched_batch.view(batch, color, height // patch_size, width // patch_size, patch_size, patch_size)
    image_batch = image_batch.permute(0, 3, 1, 4, 2, 5)
    image_batch = image_batch.contiguous().view(batch, color, height, width)
    return image_batch

def process_image(path, process_size, out_size, color):
    if color:
        # Reads image, normalizes, coverts to RGB
        image = cv2.imread(path)[100:400, 100:400, ::-1]
        image = image / 255 * 2 - 1

        # Resizes image and permutes image to (Color, Height, Width)
        processed_image = torch.tensor(cv2.resize(image, (process_size, process_size)), dtype=torch.float32)
        processed_image = processed_image.permute(2, 0, 1)
        processed_image = processed_image.contiguous()

        # Convert image to tensor and permute image to (Color, Height, Width)
        image = torch.tensor(cv2.resize(image, (out_size, out_size)), dtype=torch.float32)
        image = image.permute(2, 0, 1)
        image = image.contiguous()

    else:
        # Reads image, normalizes, converts to grayscale
        image = (cv2.imread(path, cv2.IMREAD_GRAYSCALE)[100:400, 100:400])
        image = image / 255 * 2 - 1

        # Resizes image and converts to tensor
        processed_image =  torch.tensor(cv2.resize(image, (process_size, process_size)), dtype=torch.float32)

        # Converts original image to tensor, reshapes both images to (1, Height, Width)
        processed_image = processed_image.unsqueeze(0)
        image = torch.tensor(cv2.resize(image, (out_size, out_size)), dtype=torch.float32).unsqueeze(0)

    return processed_image, image


# Creates array of image directories
def generate_directory_list(filepath, samples):
    directories = []
    for root, dirs, files in tqdm.tqdm(os.walk(filepath)):
        for file in files:
            path = os.path.join(root, file)
            path = path.replace("/", "\\")
            directories.append(path)
            if len(directories) > samples:
                break
    return directories


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

    def __init__(self, filepath = 'Datasets/Cartoon/Train', in_size = 64, out_size = 128, color = False, samples=90000):
        self.directories = generate_directory_list(filepath, samples)
        self.in_size = in_size
        self.out_size = out_size
        self.color = color

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, idx):
        image = self.directories[idx]
        return process_image(image, self.in_size, self.out_size, self.color)

# data = UpscaleDataset(color = False, samples = 100)
# im = data.__getitem__(0)[1]
# im = im.permute(1, 2, 0)
# plt.imshow(im, cmap='gray')
# plt.show()
#
# im = data.__getitem__(0)[0]
# im = im.permute(1, 2, 0)
# plt.imshow(im, cmap='gray')
# plt.show()

# load = DataLoader(data, batch_size=4, shuffle=True)
#
# for batch, _ in load:
#     print(batch.shape)
#     print(to_patches(batch, 8).shape, '\n')
#
# data = UpscaleDataset(color = True, samples = 100)
# load = DataLoader(data, batch_size=4, shuffle=True)
#
# for batch, _ in load:
#     print(batch.shape)
#     print(to_patches(batch, 8).shape, '\n')