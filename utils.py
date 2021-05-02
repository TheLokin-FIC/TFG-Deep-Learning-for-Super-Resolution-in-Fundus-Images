import os
import torch
import shutil


def remove_folder(folder):
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath) or os.path.islink(filepath):
            os.unlink(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath)


def load_checkpoint(model, optimizer, file):
    if os.path.isfile(file):
        print("[*] Loading checkpoint '" + file + "'.")
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint["model"])
        print("[*] Loaded checkpoint '" + file +
              "' (epoch " + str(epoch) + ").")

        return epoch
    else:
        print("[!] No checkpoint found at '" + file + "'.")

        return 0
