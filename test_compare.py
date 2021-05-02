import os
import cv2
import csv
import math
import lpips
import torch
import argparse
import numpy as np
import torchvision.utils as utils
import torchvision.transforms as transforms

from tqdm import tqdm
from models import Generator
from torch.utils.data import DataLoader
from dataset import TestDatasetFromFolder
from utils import remove_folder, structural_sim
from sewar.full_ref import mse, rmse, psnr, msssim


parser = argparse.ArgumentParser(
    description="Super-Resolution test comparing two upscale factors in fundus imaging.")
parser.add_argument("--dataset", type=str, metavar="N",
                    help="Folder with the dataset images.")
parser.add_argument("--model", type=str, metavar="N",
                    help="Location of the trained model.")
parser.add_argument("--crop-size", type=int, default=200, metavar="N",
                    help="Crop size for the training images (default: 200).")
parser.add_argument("--upscale-factor", type=int, default=2, metavar="N",
                    help="Low to high resolution scaling factor (default: 2).")
parser.add_argument("--upscale-factor-compare", type=int, default=4, metavar="N",
                    help="Low to high resolution scaling factor to compare (default: 4).")
opt = parser.parse_args()

target_size = opt.crop_size * opt.upscale_factor_compare
upscales = int(math.log(opt.upscale_factor_compare, opt.upscale_factor))

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

# Load dataset
dataset = TestDatasetFromFolder(
    opt.dataset, target_size, opt.upscale_factor_compare)
dataloader = DataLoader(dataset, pin_memory=True)

# Construct network architecture model of generator
model = Generator(16, opt.upscale_factor).to(device)
checkpoint = torch.load(opt.model, map_location=device)
model.load_state_dict(checkpoint["model"])

# Set the model to eval mode
model.eval()

# Reference sources from 'https://github.com/richzhang/PerceptualSimilarity'
lpips_loss = lpips.LPIPS(net="vgg").to(device)

# Algorithm performance
total_mse_value = [0, 0, 0, 0]
total_rmse_value = [0, 0, 0, 0]
total_psnr_value = [0, 0, 0, 0]
total_ssim_value = [0, 0, 0, 0]
total_l_value = [0, 0, 0, 0]
total_c_value = [0, 0, 0, 0]
total_s_value = [0, 0, 0, 0]
total_ms_ssim_value = [0, 0, 0, 0]
total_lpips_value = [0, 0, 0, 0]

# Start evaluate model performance
progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
for i, (input, target) in progress_bar:
    lr = input.to(device)
    hr = target.to(device)

    with torch.no_grad():
        sr = lr
        for _ in range(upscales):
            sr = model(sr)

    utils.save_image(lr, os.path.join(
        "test", "test_" + str(i + 1) + "_lr.bmp"))
    utils.save_image(hr, os.path.join(
        "test", "test_" + str(i + 1) + "_hr.bmp"))
    utils.save_image(sr, os.path.join(
        "test", "test_" + str(i + 1) + "_sr.bmp"))

    lr_img = cv2.imread(os.path.join(
        "test", "test_" + str(i + 1) + "_lr.bmp"))
    dst_img = cv2.imread(os.path.join(
        "test", "test_" + str(i + 1) + "_hr.bmp"))
    src_img = cv2.imread(os.path.join(
        "test", "test_" + str(i + 1) + "_sr.bmp"))

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    l_map, c_map, s_map = structural_sim(
        src_img, dst_img, win_size=11, multichannel=True, gaussian_weights=True)
    ssim_value = np.mean(l_map * c_map * s_map)
    l_value = np.mean(l_map)
    c_value = np.mean(c_map)
    s_value = np.mean(s_map)
    ms_ssim_value = msssim(src_img, dst_img)
    lpips_value = lpips_loss(sr, hr)

    total_mse_value[0] += mse_value
    total_rmse_value[0] += rmse_value
    total_psnr_value[0] += psnr_value
    total_ssim_value[0] += ssim_value
    total_l_value[0] += l_value
    total_c_value[0] += c_value
    total_s_value[0] += s_value
    total_ms_ssim_value[0] += ms_ssim_value.real
    total_lpips_value[0] += lpips_value.item()

    src_img = lr_img
    size = opt.crop_size * opt.upscale_factor
    for _ in range(upscales):
        src_img = cv2.resize(src_img, (size, size),
                             interpolation=cv2.INTER_NEAREST)
        size *= opt.upscale_factor

    cv2.imwrite(os.path.join("test", "test_" +
                             str(i + 1) + "_nn.bmp"), src_img)

    sr = transforms.ToTensor()(src_img).unsqueeze(0)
    sr = sr.to(device)

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    l_map, c_map, s_map = structural_sim(
        src_img, dst_img, win_size=11, multichannel=True, gaussian_weights=True)
    ssim_value = np.mean(l_map * c_map * s_map)
    l_value = np.mean(l_map)
    c_value = np.mean(c_map)
    s_value = np.mean(s_map)
    ms_ssim_value = msssim(src_img, dst_img)
    lpips_value = lpips_loss(sr, hr)

    total_mse_value[1] += mse_value
    total_rmse_value[1] += rmse_value
    total_psnr_value[1] += psnr_value
    total_ssim_value[1] += ssim_value
    total_l_value[1] += l_value
    total_c_value[1] += c_value
    total_s_value[1] += s_value
    total_ms_ssim_value[1] += ms_ssim_value.real
    total_lpips_value[1] += lpips_value.item()

    src_img = lr_img
    size = opt.crop_size * opt.upscale_factor
    for _ in range(upscales):
        src_img = cv2.resize(src_img, (size, size),
                             interpolation=cv2.INTER_LINEAR)
        size *= opt.upscale_factor

    cv2.imwrite(os.path.join("test", "test_" +
                             str(i + 1) + "_bl.bmp"), src_img)

    sr = transforms.ToTensor()(src_img).unsqueeze(0)
    sr = sr.to(device)

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    l_map, c_map, s_map = structural_sim(
        src_img, dst_img, win_size=11, multichannel=True, gaussian_weights=True)
    ssim_value = np.mean(l_map * c_map * s_map)
    l_value = np.mean(l_map)
    c_value = np.mean(c_map)
    s_value = np.mean(s_map)
    ms_ssim_value = msssim(src_img, dst_img)
    lpips_value = lpips_loss(sr, hr)

    total_mse_value[2] += mse_value
    total_rmse_value[2] += rmse_value
    total_psnr_value[2] += psnr_value
    total_ssim_value[2] += ssim_value
    total_l_value[2] += l_value
    total_c_value[2] += c_value
    total_s_value[2] += s_value
    total_ms_ssim_value[2] += ms_ssim_value.real
    total_lpips_value[2] += lpips_value.item()

    src_img = lr_img
    size = opt.crop_size * opt.upscale_factor
    for _ in range(upscales):
        src_img = cv2.resize(src_img, (size, size),
                             interpolation=cv2.INTER_CUBIC)
        size *= opt.upscale_factor

    cv2.imwrite(os.path.join("test", "test_" +
                             str(i + 1) + "_bc.bmp"), src_img)

    sr = transforms.ToTensor()(src_img).unsqueeze(0)
    sr = sr.to(device)

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    l_map, c_map, s_map = structural_sim(
        src_img, dst_img, win_size=11, multichannel=True, gaussian_weights=True)
    ssim_value = np.mean(l_map * c_map * s_map)
    l_value = np.mean(l_map)
    c_value = np.mean(c_map)
    s_value = np.mean(s_map)
    ms_ssim_value = msssim(src_img, dst_img)
    lpips_value = lpips_loss(sr, hr)

    total_mse_value[3] += mse_value
    total_rmse_value[3] += rmse_value
    total_psnr_value[3] += psnr_value
    total_ssim_value[3] += ssim_value
    total_l_value[3] += l_value
    total_c_value[3] += c_value
    total_s_value[3] += s_value
    total_ms_ssim_value[3] += ms_ssim_value.real
    total_lpips_value[3] += lpips_value.item()

    progress_bar.set_description(
        "[" + str(i + 1) + "/" + str(len(dataloader)) + "]")

avg_mse_value = total_mse_value[0] / len(dataloader)
avg_rmse_value = total_rmse_value[0] / len(dataloader)
avg_psnr_value = total_psnr_value[0] / len(dataloader)
avg_ssim_value = total_ssim_value[0] / len(dataloader)
avg_l_value = total_l_value[0] / len(dataloader)
avg_c_value = total_c_value[0] / len(dataloader)
avg_s_value = total_s_value[0] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[0] / len(dataloader)
avg_lpips_value = total_lpips_value[0] / len(dataloader)

with open(os.path.join("test", "results.csv"), "w+") as file:
    writer = csv.writer(file)
    writer.writerow(["network", "results"])
    writer.writerow(["Avg MSE", "{:.4f}".format(avg_mse_value)])
    writer.writerow(["Avg RMSE", "{:.4f}".format(avg_rmse_value)])
    writer.writerow(["Avg PSNR", "{:.4f}".format(avg_psnr_value)])
    writer.writerow(["Avg SSIM", "{:.4f}".format(avg_ssim_value)])
    writer.writerow(["Avg SSIM Luminance", "{:.4f}".format(avg_l_value)])
    writer.writerow(["Avg SSIM Contrast", "{:.4f}".format(avg_c_value)])
    writer.writerow(["Avg SSIM Structural", "{:.4f}".format(avg_s_value)])
    writer.writerow(["Avg MS-SSIM", "{:.4f}".format(avg_ms_ssim_value)])
    writer.writerow(["Avg LPIPS", "{:.4f}".format(avg_lpips_value)])

avg_mse_value = total_mse_value[1] / len(dataloader)
avg_rmse_value = total_rmse_value[1] / len(dataloader)
avg_psnr_value = total_psnr_value[1] / len(dataloader)
avg_ssim_value = total_ssim_value[1] / len(dataloader)
avg_l_value = total_l_value[1] / len(dataloader)
avg_c_value = total_c_value[1] / len(dataloader)
avg_s_value = total_s_value[1] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[1] / len(dataloader)
avg_lpips_value = total_lpips_value[1] / len(dataloader)

with open(os.path.join("test", "results.csv"), "a+") as file:
    writer = csv.writer(file)
    writer.writerow(["nn", "results"])
    writer.writerow(["Avg MSE", "{:.4f}".format(avg_mse_value)])
    writer.writerow(["Avg RMSE", "{:.4f}".format(avg_rmse_value)])
    writer.writerow(["Avg PSNR", "{:.4f}".format(avg_psnr_value)])
    writer.writerow(["Avg SSIM", "{:.4f}".format(avg_ssim_value)])
    writer.writerow(["Avg SSIM Luminance", "{:.4f}".format(avg_l_value)])
    writer.writerow(["Avg SSIM Contrast", "{:.4f}".format(avg_c_value)])
    writer.writerow(["Avg SSIM Structural", "{:.4f}".format(avg_s_value)])
    writer.writerow(["Avg MS-SSIM", "{:.4f}".format(avg_ms_ssim_value)])
    writer.writerow(["Avg LPIPS", "{:.4f}".format(avg_lpips_value)])

avg_mse_value = total_mse_value[2] / len(dataloader)
avg_rmse_value = total_rmse_value[2] / len(dataloader)
avg_psnr_value = total_psnr_value[2] / len(dataloader)
avg_ssim_value = total_ssim_value[2] / len(dataloader)
avg_l_value = total_l_value[2] / len(dataloader)
avg_c_value = total_c_value[2] / len(dataloader)
avg_s_value = total_s_value[2] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[2] / len(dataloader)
avg_lpips_value = total_lpips_value[2] / len(dataloader)

with open(os.path.join("test", "results.csv"), "a+") as file:
    writer = csv.writer(file)
    writer.writerow(["bl", "results"])
    writer.writerow(["Avg MSE", "{:.4f}".format(avg_mse_value)])
    writer.writerow(["Avg RMSE", "{:.4f}".format(avg_rmse_value)])
    writer.writerow(["Avg PSNR", "{:.4f}".format(avg_psnr_value)])
    writer.writerow(["Avg SSIM", "{:.4f}".format(avg_ssim_value)])
    writer.writerow(["Avg SSIM Luminance", "{:.4f}".format(avg_l_value)])
    writer.writerow(["Avg SSIM Contrast", "{:.4f}".format(avg_c_value)])
    writer.writerow(["Avg SSIM Structural", "{:.4f}".format(avg_s_value)])
    writer.writerow(["Avg MS-SSIM", "{:.4f}".format(avg_ms_ssim_value)])
    writer.writerow(["Avg LPIPS", "{:.4f}".format(avg_lpips_value)])

avg_mse_value = total_mse_value[3] / len(dataloader)
avg_rmse_value = total_rmse_value[3] / len(dataloader)
avg_psnr_value = total_psnr_value[3] / len(dataloader)
avg_ssim_value = total_ssim_value[3] / len(dataloader)
avg_l_value = total_l_value[3] / len(dataloader)
avg_c_value = total_c_value[3] / len(dataloader)
avg_s_value = total_s_value[3] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[3] / len(dataloader)
avg_lpips_value = total_lpips_value[3] / len(dataloader)

with open(os.path.join("test", "results.csv"), "a+") as file:
    writer = csv.writer(file)
    writer.writerow(["bc", "results"])
    writer.writerow(["Avg MSE", "{:.4f}".format(avg_mse_value)])
    writer.writerow(["Avg RMSE", "{:.4f}".format(avg_rmse_value)])
    writer.writerow(["Avg PSNR", "{:.4f}".format(avg_psnr_value)])
    writer.writerow(["Avg SSIM", "{:.4f}".format(avg_ssim_value)])
    writer.writerow(["Avg SSIM Luminance", "{:.4f}".format(avg_l_value)])
    writer.writerow(["Avg SSIM Contrast", "{:.4f}".format(avg_c_value)])
    writer.writerow(["Avg SSIM Structural", "{:.4f}".format(avg_s_value)])
    writer.writerow(["Avg MS-SSIM", "{:.4f}".format(avg_ms_ssim_value)])
    writer.writerow(["Avg LPIPS", "{:.4f}".format(avg_lpips_value)])

print("[*] Test - Compare done!")
