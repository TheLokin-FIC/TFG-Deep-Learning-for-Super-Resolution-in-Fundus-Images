import os
import sys
import csv
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils

from tqdm import tqdm
from models import Generator
from torch.utils.data import DataLoader
from dataset import TrainDatasetFromFolder
from utils import remove_folder, load_checkpoint

parser = argparse.ArgumentParser(
    description="Super-Resolution training on single fundus image.")
parser.add_argument("--dataset", type=str, metavar="N",
                    help="Folder with the dataset images.")
parser.add_argument("--crop-size", type=int, default=200, metavar="N",
                    help="Crop size for the training images (default: 200).")
parser.add_argument("--upscale-factor", type=int, default=2, metavar="N",
                    help="Low to high resolution scaling factor (default: 2).")
parser.add_argument("--epoch", type=int, default=5000, metavar="N",
                    help="The number of iterations for training the model (default: 5000).")
opt = parser.parse_args()

target_size = opt.crop_size * opt.upscale_factor

# Create the necessary folders
for path in [os.path.join("weight", "SRResNet"),
             os.path.join("output", "SRResNet")]:
    if not os.path.exists(path):
        os.makedirs(path)

# Show warning message
if not torch.cuda.is_available():
    device = "cpu"
    print("[!] Using CPU.")
else:
    device = "cuda:0"

# Load dataset
dataset = TrainDatasetFromFolder(opt.dataset, target_size, opt.upscale_factor)
dataloader = DataLoader(dataset, pin_memory=True)

# Construct network architecture model of generator
netG = Generator(16, opt.upscale_factor).to(device)

# Set the model to training mode
netG.train()

# Loss for SRResNet
mse_criterion = nn.MSELoss().to(device)

# Define SRResNet model optimizer
optimizer = optim.Adam(netG.parameters())

# Load SRResNet model
print("[*] Start training SRResNet model.")
checkpoint = load_checkpoint(netG, optimizer, os.path.join(
    "weight", "SRResNet", "SRResNet_" + str(opt.upscale_factor) + "x.pth"))

# Create log
if checkpoint == 0:
    with open(os.path.join("weight", "SRResNet", "SRResNet_Loss_" + str(opt.upscale_factor) + "x.csv"), "w+") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "MSE Loss"])

# Train generator using MSE loss
for epoch in range(checkpoint + 1, opt.epoch + 1):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    avg_loss = 0

    for i, (input, target) in progress_bar:
        # Set generator gradients to zero
        netG.zero_grad()

        # Generate data
        lr = input.to(device)
        hr = target.to(device)

        # Generating fake high resolution images from real low resolution images
        sr = netG(lr)

        # The MSE of the generated fake high-resolution image and real high-resolution image is calculated
        loss = mse_criterion(sr, hr)

        # Calculate gradients for generator
        loss.backward()

        # Update generator weights
        optimizer.step()

        # Show loss
        avg_loss += loss.item()
        progress_bar.set_description("[" + str(epoch) + "/" + str(opt.epoch) + "][" + str(
            i + 1) + "/" + str(len(dataloader)) + "] MSE Loss: {:.6f}".format(loss.item()))

        # An image is saved every 5000 iterations
        total_iter = i + (epoch - 1) * len(dataloader)
        if (total_iter + 1) % 5000 == 0:
            utils.save_image(lr, os.path.join(
                "output", "SRResNet", "SRResNet_" + str(total_iter + 1) + "_lr.bmp"))
            utils.save_image(hr, os.path.join(
                "output", "SRResNet", "SRResNet_" + str(total_iter + 1) + "_hr.bmp"))
            utils.save_image(sr, os.path.join(
                "output", "SRResNet", "SRResNet_" + str(total_iter + 1) + "_sr.bmp"))

    # The model is saved every epoch
    torch.save({"epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "model": netG.state_dict()
                }, os.path.join("weight", "SRResNet", "SRResNet_" + str(opt.upscale_factor) + "x.pth"))

    # Save training log
    with open(os.path.join("weight", "SRResNet", "SRResNet_Loss_" + str(opt.upscale_factor) + "x.csv"), "a+") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, avg_loss / len(dataloader)])

print("[*] Training SRResNet model done!")
