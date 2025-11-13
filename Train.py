import math

from UpscalingImages import Upscaling
from Data import UpscaleDataset
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Used strictly for development to see the images
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train(epochs=10000, lr=0.0001, save_every=1, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Loads model from filepath or creates new model
    model, epoch = Upscaling.load("Models/compress_model.pt")
    model = model.to(device)

    # Create Dataset and send into dataloader
    dataset = UpscaleDataset(color = True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initiate Loss and Optimizer functions
    loss_fn = nn.MSELoss()
    adam = optim.Adam(model.parameters(), lr=lr)

    # Loop through epochs, starting at 0 or where model left off
    for i in range(epoch, epochs):
        progress_bar = tqdm(loader, desc=f"Epoch {i}/{epochs}")
        running_loss = 0.0

        for j, (batch_input, batch_target) in enumerate(progress_bar):
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            # Find values
            adam.zero_grad()
            output = model(batch_input)

            # Compute Loss
            loss = loss_fn(output, batch_target)
            running_loss += loss.item()

            # Update weights
            loss.backward()
            adam.step()

            # Display Loss and PSNR
            PSNR = 10 * math.log10(1 / loss.item())
            progress_bar.set_postfix({"Loss:": loss.item(), "PSNR:": PSNR})

        # Save model every __ epochs
        if i % save_every == 0:
            print(f"Saving model at epoch {i}")
            Upscaling.save(model, "Models/compress_model.pt", i)

        if 0.01 > (running_loss / len(loader)):
            compression = max(50, dataset.get_compression() - 10)
            if compression > 50: print(f'Lowering Compression: {compression}')
            dataset.set_compression(compression)


def validate(test_loader= 'Datasets/Cartoon/Test', samples = 10000, compress = 100):
    dataset = UpscaleDataset(samples=samples, color = True, filepath = test_loader)
    dataset.set_compression(compress)
    current, epoch = Upscaling.load("Models/curriculum_model.pt")
    lossy, epoch = Upscaling.load("Models/lossy_model.pt")

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_input, batch_target in loader:
        with torch.no_grad():
            curriculum_output = current(batch_input)
            lossy_output= lossy(batch_input)
            figs, axs = plt.subplots(2, 3)
            for i in range(2):
                axs[i][0].set_title("Transformer")
                axs[i][1].set_title("Original")
                axs[i][2].set_title("Input")

                axs[i][1].imshow(batch_target[0].detach().permute(1, 2, 0))
                axs[i][2].imshow(batch_input[0].detach().permute(1, 2, 0))
                for j in range(3):
                    axs[i][j].axis("off")

            axs[0][0].imshow(curriculum_output[0].detach().permute(1, 2, 0))
            axs[1][0].imshow(lossy_output[0].detach().permute(1, 2, 0))

            plt.show()

# train()
validate()