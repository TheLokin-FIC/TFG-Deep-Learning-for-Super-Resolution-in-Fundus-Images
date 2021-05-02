import os
import cv2
import errno
import shutil
import random
import numpy as np


def remove_edges(img_source, threthold_low=7, threthold_high=180):
    if isinstance(img_source, str):
        img = cv2.imread(img_source)
    else:
        img = img_source
    if img is None:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), img_source)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    width, height = (img.shape[1], img.shape[0])

    (left, bottom) = (0, 0)
    (right, top) = (img.shape[1], img.shape[0])

    for i in range(width):
        array = img[:, i, :]
        if np.sum(array) > threthold_low * array.shape[0] * array.shape[1] and np.sum(array) < threthold_high * array.shape[0] * array.shape[1]:
            left = i
            break
    left = max(0, left)

    for i in range(width - 1, 0 - 1, -1):
        array = img[:, i, :]
        if np.sum(array) > threthold_low * array.shape[0] * array.shape[1] and np.sum(array) < threthold_high * array.shape[0] * array.shape[1]:
            right = i
            break
    right = min(width, right)

    for i in range(height):
        array = img[i, :, :]
        if np.sum(array) > threthold_low * array.shape[0] * array.shape[1] and np.sum(array) < threthold_high * array.shape[0] * array.shape[1]:
            bottom = i
            break
    bottom = max(0, bottom)

    for i in range(height - 1, 0 - 1, -1):
        array = img[i, :, :]
        if np.sum(array) > threthold_low * array.shape[0] * array.shape[1] and np.sum(array) < threthold_high * array.shape[0] * array.shape[1]:
            top = i
            break
    top = min(height, top)

    return img[bottom:top, left:right, :]


def crop_image(img_source):
    if isinstance(img_source, str):
        img = cv2.imread(img_source)
    else:
        img = img_source
    if img is None:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), img_source)

    width, height = (img.shape[1], img.shape[0])
    minWidthHeight = min(width, height)

    minRadius = round(minWidthHeight * 0.33)
    maxRadius = round(minWidthHeight * 0.6)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450,
                               param1=120, param2=32, minRadius=minRadius, maxRadius=maxRadius)

    (x, y, r) = (0, 0, 0)
    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if circles is not None:
            x1, y1, _ = circles[0]
            if x1 > (2 / 5 * width) and x1 < (3 / 5 * width) and y1 > (2 / 5 * height) and y1 < (3 / 5 * height):
                x, y, r = circles[0]
                found_circle = True

    if not found_circle:
        # Suppose the center of the image is the center of the circle
        x = img.shape[1] // 2
        y = img.shape[0] // 2

        # Get radius according to the distribution of pixels of the middle line
        temp_x = img[int(img.shape[0] / 2), :, :].sum(1)
        r = int((temp_x > temp_x.mean() / 12).sum() / 2)

    l = 2 * (r - 2) / np.sqrt(2)
    bottom = y - int(l / 2)
    top = y + int(l / 2)
    left = x - int(l / 2)
    right = x + int(l / 2)

    return img[bottom:top, left:right, :]


def split_image(img_source, row_number=2, col_number=2):
    if isinstance(img_source, str):
        img = cv2.imread(img_source)
    else:
        img = img_source
    if img is None:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), img_source)

    img_blocks = []
    for blocks in np.array_split(img, row_number, axis=0):
        img_blocks += np.array_split(blocks, col_number, axis=1)

    return img_blocks


def split_dataset(dataset_dir):
    # The original data set is divided into 9:1 (train:val)
    for _, _, files in os.walk(dataset_dir):
        # Number of files in val set
        val_number = int(0.1 * len(files))

        # Gets a list of file names selected by random functions
        samples = random.sample(files, val_number)

        # Move the validation set to the specified location
        for filename in samples:
            shutil.move(os.path.join(dataset_dir, "train", filename),
                        os.path.join(dataset_dir, "test", filename))
