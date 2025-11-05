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


def train(epochs=1000, lr=0.001, save_every=10, loss_every=1, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = UpscaleDataset(samples = 100)
    model, epoch = Upscaling.load("model.pt")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_function = nn.MSELoss()
    adam = optim.Adam(model.parameters(), lr=lr)

    for i in range(epoch, epochs):
        progress_bar = tqdm(loader, desc=f"Epoch {i + 1}/{epochs}")
        # print(f"Epoch {i+1}")
        counter = 0
        running_loss = 0.0

        for j, batch in enumerate(progress_bar):
            batch_input, batch_target = batch[0], batch[1]
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            adam.zero_grad()
            output = model(batch_input)

            loss = loss_function(output, batch_target)
            running_loss += loss.item()

            loss.backward()
            adam.step()

            counter += 1

            if counter % loss_every == 0:
                progress_bar.set_postfix({"Loss:": loss.item()})
                total_loss = 0

        if i % save_every == 0:
            print(f"Saving model at epoch {i + 1}")
            Upscaling.save(model, "model.pt", i)


train()