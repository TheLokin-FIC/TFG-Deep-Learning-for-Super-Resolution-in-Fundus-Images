import os
import onnx
import argparse
import torch.onnx
import numpy as np

from models import Generator


parser = argparse.ArgumentParser(
    description="Converting a Super-Resolution model in Pytorch to ONNX Runtime.")
parser.add_argument("--model", type=str, metavar="N",
                    help="Location of the trained model.")
parser.add_argument("--upscale-factor", type=int, metavar="N",
                    help="Low to high resolution scaling factor.")
parser.add_argument("--output", type=str, metavar="N", default="",
                    help="Location where to save the ONNX model (default: /).")
opt = parser.parse_args()

# Selection of appropriate treatment equipment
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

# Input to the model
x = torch.randn(1, 3, 100, 100, requires_grad=True)
x = x.to(device)
torch_out = model(x)

# Export the model with:
# - model being run
# - model input (or a tuple for multiple inputs)
# - where to save the model (or a tuple for multiple inputs)
# - store the trained parameter weights inside the model file
# - the ONNX version to export the model to
# - whether to execute constant folding for optimization
# - the model's input names
# - the model's output names
# - variable lenght axes
torch.onnx.export(model, x, os.path.join(opt.output, "super_resolution_" + str(opt.upscale_factor) + "x.onnx"), export_params=True, opset_version=10, do_constant_folding=True, input_names=[
                  'input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# Check ONNX Runtime model
onnx_model = onnx.load(os.path.join(
    opt.output, "super_resolution_" + str(opt.upscale_factor) + "x.onnx"))
onnx.checker.check_model(onnx_model)

print("[*] Converting the model in Pytorch to ONNX Runtime done!")
