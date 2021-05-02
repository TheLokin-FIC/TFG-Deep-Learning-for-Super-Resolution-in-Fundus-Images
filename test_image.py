import os
import cv2
import lpips
import torch
import argparse
import numpy as np
import torchvision.utils as utils
import torchvision.transforms as transforms

from PIL import Image
from models import Generator
from utils import remove_folder

parser = argparse.ArgumentParser(
    description="Super-Resolution test on a single fundus image.")
parser.add_argument("--image", type=str, metavar="N",
                    help="Low resolution image location.")
parser.add_argument("--model", type=str, metavar="N",
                    help="Location of the trained model.")
parser.add_argument("--upscale-factor", type=int, default=2, metavar="N",
                    help="Low to high resolution scaling factor (default: 2).")
opt = parser.parse_args()

# Create the necessary folders
if os.path.exists("test"):
    remove_folder("test")
else:
    os.makedirs("test")

# Selection of the appropriate device
if not torch.cuda.is_available():
    device = "cpu"
    print("[!] Using CPU.")
else:
    device = "cuda:0"

# Construct network architecture model of generator
model = Generator(16, opt.upscale_factor).to(device)
checkpoint = torch.load(opt.model, map_location=device)
model.load_state_dict(checkpoint["model"])

# Set the model to eval mode
model.eval()

# Load image
lr = Image.open(opt.image)
lr = transforms.ToTensor()(lr).unsqueeze(0)
lr = lr.to(device)

# Start model performance
with torch.no_grad():
    sr = model(lr)

# Save result
utils.save_image(sr, os.path.join("test", "test_sr.bmp"))

print("[*] Test - Single image done!")
