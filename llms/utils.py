import time
import torch
from typing import Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# Device Detection
def get_device() -> torch.device:
    """
    Detect the device for computation.

    Returns:
        torch.device: The detected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
#         return torch.device("mps")
        return torch.device("cpu")
    else:
        return torch.device("cpu")
    
def get_hardware_info() -> dict:
    """
    Gather hardware details about the system used for evaluation.

    Returns:
        dict: A dictionary containing the number of GPUs, GPU types, and whether CUDA is available.
    """
    device = get_device()
    if device.type == "cuda":
        nvmlInit()
        gpu_count = nvmlDeviceGetCount()
        gpu_info = [nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)).decode() for i in range(gpu_count)]
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

def evaluate_latency_throughput(
    model: Any, tokenizer: Any, prompt: str, max_tokens: int = 50, batch_size: int = 1
) -> Tuple[Any, float, float, float]:
    """
    Evaluate latency and throughput of the model.

    Args:
        model (Any): The Hugging Face model instance.
        tokenizer (Any): The Hugging Face tokenizer instance.
        prompt (str): The input prompt for evaluation.
        max_tokens (int): The maximum number of tokens to generate.
        batch_size (int): Batch size for evaluation.

    Returns:
        Tuple[Any, float, float, float]: Outputs, latency, throughput, and token throughput.
    """
    device = get_device()
    model = model.to(device)

    # Prepare inputs
    inputs = [prompt] * batch_size
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)

    # Warm-up
    print("Warming up...")
    start_warmup = time.time()
    model.generate(**tokenized_inputs, max_new_tokens=max_tokens)
    end_warmup = time.time()
    warmup_time = end_warmup - start_warmup
    print(f"Warm-up Time: {warmup_time:.2f}s")

    # Measure latency and throughput
    print("Measuring latency and throughput...")
    start_time = time.time()
    outputs = model.generate(**tokenized_inputs, max_new_tokens=max_tokens)
    end_time = time.time()

    latency = end_time - start_time
    throughput = batch_size / latency
    total_tokens = max_tokens * batch_size
    token_throughput = total_tokens / latency
    print(f"Latency: {latency:.2f}s | Throughput: {throughput:.2f} responses/sec | Token Throughput: {token_throughput:.2f} tokens/sec")
    return outputs, latency, throughput, token_throughput


def evaluate_power_efficiency(
    model: Any, tokenizer: Any, prompt: str, max_tokens: int = 50, batch_size: int = 1
) -> Tuple[float, float]:
    """
    Evaluate power efficiency (works only on CUDA devices).

    Args:
        model (Any): The Hugging Face model instance.
        tokenizer (Any): The Hugging Face tokenizer instance.
        prompt (str): The input prompt for evaluation.
        max_tokens (int): The maximum number of tokens to generate.
        batch_size (int): Batch size for evaluation.

    Returns:
        Tuple[float, float]: Power consumed and energy per token.
    """
    device = get_device()
    if device.type != "cuda":
        print("Power efficiency evaluation is only supported on CUDA devices.")
        return 0.0, 0.0

    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage

    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)

    def track_power():
        """Track GPU power consumption in watts."""
        return nvmlDeviceGetPowerUsage(gpu_handle) / 1000

    model = model.to(device)
    inputs = [prompt] * batch_size
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)

    # Warm-up
    print("Warming up...")
    model.generate(**tokenized_inputs, max_new_tokens=max_tokens)

    # Measure power and efficiency
    print("Measuring power efficiency...")
    power_start = track_power()
    start_time = time.time()

    model.generate(**tokenized_inputs, max_new_tokens=max_tokens)

    end_time = time.time()
    power_end = track_power()

    latency = end_time - start_time
    total_tokens = max_tokens * batch_size
    power_consumed = (power_end - power_start) * latency
    energy_per_token = power_consumed / total_tokens if total_tokens > 0 else float("inf")

    print(f"Power Consumption: {power_consumed:.2f} W | Energy per Token: {energy_per_token:.4f} W/token")
    return power_consumed, energy_per_token


def compare_precision_accuracy(
    model_name: str, prompt: str, max_tokens: int = 50
) -> bool:
    """
    Compare outputs for fp32 and fp16 precision to test output integrity.

    Args:
        model_name (str): The Hugging Face model name or path.
        prompt (str): The input prompt for the model.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        bool: Whether the outputs for fp32 and fp16 are identical.
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Full precision (fp32)
    model_fp32 = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_fp32 = model_fp32.generate(**inputs, max_new_tokens=max_tokens)
    text_fp32 = tokenizer.decode(output_fp32[0], skip_special_tokens=True)

    # Mixed precision (fp16)
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    output_fp16 = model_fp16.generate(**inputs, max_new_tokens=max_tokens)
    text_fp16 = tokenizer.decode(output_fp16[0], skip_special_tokens=True)

    print(f"Full Precision (fp32) Output:\n{text_fp32}")
    print(f"Mixed Precision (fp16) Output:\n{text_fp16}")

    return text_fp32 == text_fp16


def memory_by_sequence_length(
    model: Any, tokenizer: Any, prompt: str, max_tokens: int = 50, max_length: int = 1024
) -> Dict[int, float]:
    """
    Measure memory usage as sequence length increases.

    Args:
        model (Any): The Hugging Face model instance.
        tokenizer (Any): The Hugging Face tokenizer instance.
        prompt (str): The input prompt for evaluation.
        max_tokens (int): The maximum number of tokens to generate.
        max_length (int): The maximum sequence length.

    Returns:
        Dict[int, float]: Memory usage (in MB) for different sequence lengths.
    """
    device = get_device()
    model = model.to(device)
    memory_results = {}

    for seq_length in [128, 256, 512, max_length]:
        prompt_repeated = prompt * (seq_length // len(prompt))
        inputs = tokenizer(prompt_repeated, return_tensors="pt", truncation=True).to(device)

        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        model.generate(**inputs, max_new_tokens=max_tokens)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0
        memory_results[seq_length] = peak_memory

    return memory_results
