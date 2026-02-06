import torch
import numpy as np
import random
import os

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device(device_pref: str = "auto"):
    """Get the appropriate torch device with fallback support."""
    if device_pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    # Try the preferred device, fallback if unavailable
    try:
        device = torch.device(device_pref)
        # Test if available (e.g., if "cuda" requested but no GPU)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return device
    except Exception:
        return torch.device("cpu")
