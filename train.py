import os
import sys
import csv
import torch
import argparse
import pytorch_ssim
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils

from tqdm import tqdm
from loss import ContentLoss
from torch.utils.data import DataLoader
from dataset import TrainDatasetFromFolder
from models import Generator, Discriminator
from utils import remove_folder, load_checkpoint


parser = argparse.ArgumentParser(
    description="Super-Resolution training on fundus imaging.")
parser.add_argument("--dataset", type=str, metavar="N",
                    help="Folder with the dataset images.")
parser.add_argument("--architecture", type=int, metavar="N",
                    help="The architecture to use: 1. SRResnet (MSE). 2. SRRGAN. 3. SRResnet (SSIM).")
parser.add_argument("--crop-size", type=int, default=200, metavar="N",
                    help="Crop size for the training images (default: 200).")
parser.add_argument("--upscale-factor", type=int, default=2, metavar="N",
                    help="Low to high resolution scaling factor (default: 2).")
parser.add_argument("--epoch", type=int, default=5000, metavar="N",
                    help="The number of iterations for training the model (default: 5000).")
opt = parser.parse_args()

target_size = opt.crop_size * opt.upscale_factor

# Create the necessary folders
for path in [os.path.join("output"),
             os.path.join("weight", "SRResNet", "MSE"),
             os.path.join("weight", "SRGAN"),
             os.path.join("weight", "SRResNet", "SSIM")]:
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

# SRResNet (MSE)
if opt.architecture == 1:
    # Construct network architecture model of generator
    netG = Generator(16, opt.upscale_factor).to(device)

    # Set the model to training mode
    netG.train()

    # Loss SRResNet
    mse_criterion = nn.MSELoss().to(device)

    # Define SRResNet model optimizer
    optimizer = optim.Adam(netG.parameters())

    # Loading SRResNet checkpoint
    print("[*] Start training SRResNet (MSE) model.")
    checkpoint = load_checkpoint(netG, optimizer, os.path.join(
        "weight", "SRResNet", "MSE", "SRResNet_" + str(opt.upscale_factor) + "x.pth"))

    # Create SRResNet log
    if checkpoint == 0:
        with open(os.path.join("weight", "SRResNet", "MSE", "SRResNet_Loss_" + str(opt.upscale_factor) + "x.csv"), "w+") as file:
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
                    "output", "train_" + str(total_iter + 1) + "_lr.bmp"))
                utils.save_image(hr, os.path.join(
                    "output", "train_" + str(total_iter + 1) + "_hr.bmp"))
                utils.save_image(sr, os.path.join(
                    "output", "train_" + str(total_iter + 1) + "_sr.bmp"))

        # The model is saved every epoch
        torch.save({"epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "model": netG.state_dict()
                    }, os.path.join("weight", "SRResNet", "MSE", "SRResNet_" + str(opt.upscale_factor) + "x.pth"))

        # Save training log
        with open(os.path.join("weight", "SRResNet", "MSE", "SRResNet_Loss_" + str(opt.upscale_factor) + "x.csv"), "a+") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_loss / len(dataloader)])

    print("[*] Training SRResNet (MSE) model done!")

# SRGAN
elif opt.architecture == 2:
    # Construct network architecture model of generator and discriminator
    netG = Generator(16, opt.upscale_factor).to(device)
    netD = Discriminator().to(device)

    # Set the models to training mode
    netG.train()
    netD.train()

    # We use VGG as our feature extraction method by default
    content_criterion = ContentLoss().to(device)

    # Perceptual loss = content loss + 1e-3 * adversarial loss
    adversarial_criterion = nn.BCELoss().to(device)

    # Define SRGAN model optimizers
    optimizerD = optim.Adam(netD.parameters())
    optimizerG = optim.Adam(netG.parameters())
    step_size = max(1, int((opt.epoch // len(dataloader)) // 2))
    schedulerD = optim.lr_scheduler.StepLR(
        optimizerD, step_size=step_size, gamma=0.1)
    schedulerG = optim.lr_scheduler.StepLR(
        optimizerG, step_size=step_size, gamma=0.1)

    # Loading SRGAN checkpoint
    print("[*] Starting training SRGAN model")
    checkpoint = load_checkpoint(netG, optimizerG, os.path.join(
        "weight", "SRGAN", "netG_" + str(opt.upscale_factor) + "x.pth"))
    checkpoint = load_checkpoint(netD, optimizerD, os.path.join(
        "weight", "SRGAN", "netD_" + str(opt.upscale_factor) + "x.pth"))

    # Create SRGAN log
    if checkpoint == 0:
        with open(os.path.join("weight", "SRGAN", "SRGAN_Loss_" + str(opt.upscale_factor) + "x.csv"), "w+") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "D Loss", "G Loss"])

    # Train generator and discriminator
    for epoch in range(checkpoint + 1, opt.epoch + 1):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        g_avg_loss = 0
        d_avg_loss = 0

        for i, (input, target) in progress_bar:
            # Generate data
            lr = input.to(device)
            hr = target.to(device)

            batch_size = lr.size(0)
            real_label = torch.ones(
                batch_size, dtype=lr.dtype, device=device)
            fake_label = torch.zeros(
                batch_size, dtype=lr.dtype, device=device)

            ###############################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
            ###############################################################

            # Set discriminator gradients to zero
            netD.zero_grad()

            # Generate a super-resolution image
            sr = netG(lr)

            # Train with real high resolution image
            hr_output = netD(hr)
            errD_hr = adversarial_criterion(hr_output, real_label)
            errD_hr.backward()
            D_x = hr_output.mean().item()

            # Train with fake high resolution image
            sr_output = netD(sr.detach())
            errD_sr = adversarial_criterion(sr_output, fake_label)
            errD_sr.backward()
            D_G_z1 = sr_output.mean().item()
            errD = errD_hr + errD_sr
            optimizerD.step()

            ###############################################
            # (2) Update G network: maximize log(D(G(z))) #
            ###############################################

            # Set generator gradients to zero
            netG.zero_grad()

            # We then define the VGG loss as the euclidean distance between the feature representations of
            # a reconstructed image G(LR) and the reference image LR
            content_loss = content_criterion(sr, hr)

            # Second train with fake high resolution image
            sr_output = netD(sr)

            # The generative loss is defined based on the probabilities of the discriminator
            # D(G(LR)) over all training samples as
            adversarial_loss = adversarial_criterion(sr_output, real_label)

            # We formulate the perceptual loss as the weighted sum of a content loss and an adversarial loss component as
            errG = content_loss + 1e-3 * adversarial_loss
            errG.backward()
            D_G_z2 = sr_output.mean().item()
            optimizerG.step()

            # Dynamic adjustment of learning rate
            schedulerD.step()
            schedulerG.step()

            d_avg_loss += errD.item()
            g_avg_loss += errG.item()

            # Show loss
            progress_bar.set_description("[" + str(epoch) + "/" + str(opt.epoch) + "][" + str(i + 1) + "/" + str(len(dataloader)) +
                                         "] Loss_D: {:.6f} Loss_G: {:.6f} ".format(errD.item(), errG.item()) + "D(HR): {:.6f} D(G(LR)): {:.6f}/{:.6f}".format(D_x, D_G_z1, D_G_z2))

            # An image is saved every 5000 iterations
            total_iter = i + (epoch - 1) * len(dataloader)
            if (total_iter + 1) % 5000 == 0:
                utils.save_image(lr, os.path.join(
                    "output", "train_" + str(total_iter + 1) + "_lr.bmp"))
                utils.save_image(hr, os.path.join(
                    "output", "train_" + str(total_iter + 1) + "_hr.bmp"))
                utils.save_image(sr, os.path.join(
                    "output", "train_" + str(total_iter + 1) + "_sr.bmp"))

        # The models are saved every epoch
        torch.save({"epoch": epoch,
                    "optimizer": optimizerD.state_dict(),
                    "model": netD.state_dict()
                    }, os.path.join("weight", "SRGAN", "netD_" + str(opt.upscale_factor) + "x.pth"))
        torch.save({"epoch": epoch,
                    "optimizer": optimizerG.state_dict(),
                    "model": netG.state_dict()
                    }, os.path.join("weight", "SRGAN", "netG_" + str(opt.upscale_factor) + "x.pth"))

        # Save training log
        with open(os.path.join("weight", "SRGAN", "SRGAN_Loss_" + str(opt.upscale_factor) + "x.csv"), "a+") as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch, d_avg_loss / len(dataloader), g_avg_loss / len(dataloader)])

    print("[*] Training SRGAN model done!")

# SRResNet (SSIM)
elif opt.architecture == 3:
    # Construct network architecture model of generator
    netG = Generator(16, opt.upscale_factor).to(device)

    # Set the model to training mode
    netG.train()

    # Loss SRResNet
    ssim_criterion = pytorch_ssim.SSIM().to(device)

    # Define SRResNet model optimizer
    optimizer = optim.Adam(netG.parameters())

    # Loading SRResNet checkpoint
    print("[*] Start training SRResNet (SSIM) model.")
    checkpoint = load_checkpoint(netG, optimizer, os.path.join(
        "weight", "SRResNet", "SSIM", "SRResNet_" + str(opt.upscale_factor) + "x.pth"))

    # Create SRResNet log
    if checkpoint == 0:
        with open(os.path.join("weight", "SRResNet", "SSIM", "SRResNet_Loss_" + str(opt.upscale_factor) + "x.csv"), "w+") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "SSIM Loss"])

    # Train generator using SSIM loss
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

            # The SSIM of the generated fake high-resolution image and real high-resolution image is calculated
            loss = -ssim_criterion(sr, hr)

            # Calculate gradients for generator
            loss.backward()

            # Update generator weights
            optimizer.step()

            # Show loss
            avg_loss -= loss.item()
            progress_bar.set_description("[" + str(epoch) + "/" + str(opt.epoch) + "][" + str(
                i + 1) + "/" + str(len(dataloader)) + "] SSIM Loss: {:.6f}".format(-loss.item()))

            # An image is saved every 5000 iterations
            total_iter = i + (epoch - 1) * len(dataloader)
            if (total_iter + 1) % 5000 == 0:
                utils.save_image(lr, os.path.join(
                    "output", "train_" + str(total_iter + 1) + "_lr.bmp"))
                utils.save_image(hr, os.path.join(
                    "output", "train_" + str(total_iter + 1) + "_hr.bmp"))
                utils.save_image(sr, os.path.join(
                    "output", "train_" + str(total_iter + 1) + "_sr.bmp"))

        # The model is saved every epoch
        torch.save({"epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "model": netG.state_dict()
                    }, os.path.join("weight", "SRResNet", "SSIM", "SRResNet_" + str(opt.upscale_factor) + "x.pth"))

        # Save training log
        with open(os.path.join("weight", "SRResNet", "SSIM", "SRResNet_Loss_" + str(opt.upscale_factor) + "x.csv"), "a+") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_loss / len(dataloader)])

    print("[*] Training SRResNet (SSIM) model done!")

else:
    print("[!] Wrong architecture.")
