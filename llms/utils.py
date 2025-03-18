import time
import torch
from typing import Tuple, Dict, Any, List
from transformers import PreTrainedModel, PreTrainedTokenizer

# Optional, for NVIDIA GPU power monitoring.  Raises an error if pynvml is not available.
try:
    from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetName, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("pynvml not found. Power monitoring will not be available. "
          "You can install it with 'pip install nvidia-ml-py'.")



def get_device() -> torch.device:
    """
    Detects the available computation device.

    Returns:
        torch.device: The detected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Corrected: Enable MPS if available
    else:
        return torch.device("cpu")


def get_hardware_info() -> Dict[str, Any]:
    """
    Gathers hardware details about the system.

    Returns:
        Dict[str, Any]: A dictionary containing the number of GPUs, GPU types, and the device type.
    """
    device = get_device()
    if device.type == "cuda":
        if NVML_AVAILABLE:  # Check if NVML is available before using it
            nvmlInit()
            gpu_count = nvmlDeviceGetCount()
            gpu_info = [nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)).decode() for i in range(gpu_count)]
        else:
            gpu_count = 0  # Set to 0 if NVML isn't available
            gpu_info = ["N/A (pynvml not available)"]  # Provide a message
        return {
            "num_gpus": gpu_count,
            "gpu_types": gpu_info,
            "device": device.type,
        }
    elif device.type == "mps":
        return {
            "num_gpus": 1,
            "gpu_types": ["Apple M1/M2/M3 (Metal Performance Shaders)"],
            "device": device.type,
        }
    else:
        return {
            "num_gpus": 0,
            "gpu_types": ["CPU"],
            "device": device.type,
        }