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


def train(epochs=300, lr=0.0001, save_every=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Loads model from filepath or creates new model
    model, epoch = Upscaling.load("model.pt")
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
            Upscaling.save(model, "model.pt", i)

def validate(test_loader= 'Datasets/Cartoon/Test', samples = 10000):
    dataset = UpscaleDataset(samples=samples, filepath = test_loader)
    model, epoch = Upscaling.load("model3.pt")
    # cnn_model, epoch2 = CNNModel.load("cnn_model.pt")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_input, batch_target in loader:
        with torch.no_grad():
            output = model(batch_input)
            # output2 = cnn_model(batch_input)
            figs, axs = plt.subplots(1, 2)
            axs[0].axis("off")
            axs[1].axis("off")
            # axs[2].axis("off")

            axs[0].set_title("Transformer")
            axs[1].set_title("Original")
            # axs[2].set_title("CNN")

            axs[0].imshow(output[0].cpu().detach().squeeze(), cmap="gray")
            axs[1].imshow(batch_target[0].cpu().detach().squeeze(), cmap="gray")
            # axs[2].imshow(output2[0].cpu().detach().squeeze(), cmap="gray")
            plt.show()


train()
# validate()