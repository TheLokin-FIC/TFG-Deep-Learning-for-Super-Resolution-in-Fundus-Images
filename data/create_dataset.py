import os
import cv2
import random
import argparse

from tqdm import tqdm
from utils import remove_edges, crop_image


parser = argparse.ArgumentParser(
    description="Create a new dataset from another.")
parser.add_argument("--dataset", type=str, metavar="N",
                    help="Folder with the dataset images.")
parser.add_argument("--output", type=str, default=".", metavar="N",
                    help="Folder for the new dataset (default: .).")
opt = parser.parse_args()

# Preprocess dataset
for filename in tqdm(os.listdir(opt.dataset), desc="Generating images from dataset"):
    img = cv2.imread(os.path.join(opt.dataset, filename))
    img = remove_edges(img)
    img = crop_image(img)

    cv2.imwrite(os.path.join(opt.output, filename), img)
