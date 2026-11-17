import numpy as np
import xarray as xr

import torch

from aurora import Batch, Metadata
from aurora.normalisation import locations, scales, normalise_atmos_var, normalise_surf_var

import yaml

import subprocess

def memcheck():
    """Print GPU memory usage statistics."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.1f} MiB")
            print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024 ** 2:.1f} MiB")
            print(f"  Max Allocated: {torch.cuda.max_memory_allocated(i) / 1024 ** 2:.1f} MiB")
            print(f"  Max Reserved:  {torch.cuda.max_memory_reserved(i) / 1024 ** 2:.1f} MiB") 
    else:
        print("CUDA not available")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def smart_two_sigfigs(x, pos):
    # Use scientific notation for very small or large numbers
    if x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e3):
        return f"{x:.2e}"  # e.g., 1.2e-03
    else:
        # Otherwise, use up to 2 decimals, trimming zeros
        return f"{x:.2f}".rstrip('0').rstrip('.')
    
def gif_to_mp4(gif_path, mp4_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", gif_path,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        "-movflags", "faststart",
        mp4_path
    ]
    subprocess.run(cmd, check=True)

def lat_formatter(x, pos):
    if x > 0:
        return f"{x:.0f}°N"
    elif x < 0:
        return f"{abs(x):.0f}°S"
    else:
        return "0°"