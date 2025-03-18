import time
import torch
from typing import Tuple, Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from utils import get_device, NVML_AVAILABLE  # Import from utils.py

if NVML_AVAILABLE:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage


def evaluate_latency_throughput(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_tokens: int = 50,
    batch_size: int = 1
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Evaluates the latency and throughput of a given Hugging Face model.

    Args:
        model (PreTrainedModel): The Hugging Face model instance.
        tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer instance.
        prompt (str): The input prompt for evaluation.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 50.
        batch_size (int): The batch size for evaluation. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, float, float, float]: A tuple containing the model's outputs,
                                                latency (in seconds),
                                                throughput (responses/second), and
                                                token throughput (tokens/second).
    """
    device = get_device()
    model = model.to(device)  # Move model to the correct device

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
    total_tokens = max_tokens * batch_size  # Correct calculation for generated tokens.
    token_throughput = total_tokens / latency
    print(f"Latency: {latency:.2f}s | Throughput: {throughput:.2f} responses/sec | Token Throughput: {token_throughput:.2f} tokens/sec")
    return outputs, latency, throughput, token_throughput


def evaluate_power_efficiency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_tokens: int = 50,
    batch_size: int = 1
) -> Tuple[float, float]:
    """
    Evaluates the power efficiency of a given Hugging Face model (CUDA devices only).

    Args:
        model (PreTrainedModel): The Hugging Face model instance.
        tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer instance.
        prompt (str): The input prompt for evaluation.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 50.
        batch_size (int): The batch size for evaluation. Defaults to 1.

    Returns:
        Tuple[float, float]: A tuple containing the total power consumed (in Watts)
                            and energy per token (in Watts/token).  Returns (0.0, 0.0)
                            if not on a CUDA device or if pynvml is not available.
    """
    device = get_device()
    if device.type != "cuda" or not NVML_AVAILABLE:
        print("Power efficiency evaluation is only supported on CUDA devices with pynvml installed.")
        return 0.0, 0.0

    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)

    def track_power() -> float:
        """Tracks GPU power consumption in watts."""
        return nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0  # Convert milliwatts to watts

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
    power_consumed = (power_end + power_start)/2 * latency  # More accurate: average power during generation

    # Avoid division by zero
    energy_per_token = power_consumed / total_tokens if total_tokens > 0 else 0.0

    print(f"Power Consumption: {power_consumed:.2f} W | Energy per Token: {energy_per_token:.6f} W/token")
    return power_consumed, energy_per_token


def compare_precision_accuracy(
    model_name: str,
    prompt: str,
    max_tokens: int = 50
) -> bool:
    """
    Compares outputs for fp32 and fp16 precision to assess output integrity.

    Args:
        model_name (str): The Hugging Face model name or path.
        prompt (str): The input prompt for the model.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 50.

    Returns:
        bool: True if the outputs for fp32 and fp16 are identical, False otherwise.
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Full precision (fp32)
    model_fp32 = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_fp32 = model_fp32.generate(**inputs, max_new_tokens=max_tokens)
    text_fp32 = tokenizer.decode(output_fp32[0], skip_special_tokens=True)

    # Mixed precision (fp16) - Use autocast for best practice
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

    with torch.cuda.amp.autocast():  # Use autocast for mixed precision
        output_fp16 = model_fp16.generate(**inputs, max_new_tokens=max_tokens)
    text_fp16 = tokenizer.decode(output_fp16[0], skip_special_tokens=True)


    print(f"Full Precision (fp32) Output:\n{text_fp32}")
    print(f"Mixed Precision (fp16) Output:\n{text_fp16}")

    return text_fp32 == text_fp16



def memory_by_sequence_length(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_tokens: int = 50,
    max_length: int = 1024
) -> Dict[int, float]:
    """
    Measures memory usage as the input sequence length increases.

    Args:
        model (PreTrainedModel): The Hugging Face model instance.
        tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer instance.
        prompt (str): The input prompt for evaluation.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 50.
        max_length (int): The maximum sequence length to test. Defaults to 1024.

    Returns:
        Dict[int, float]: A dictionary mapping sequence lengths (int) to
                         peak memory usage (in MB).  Returns 0.0 for non-CUDA devices.
    """
    device = get_device()
    model = model.to(device)
    memory_results: Dict[int, float] = {}

    for seq_length in [128, 256, 512, max_length]:
        print(f"Measuring memory usage for sequence length: {seq_length}")
        # Create a prompt that, when tokenized, will be approximately seq_length long
        prompt_repeated = prompt * (seq_length // len(tokenizer.encode(prompt)) + 1)
        inputs = tokenizer(prompt_repeated, return_tensors="pt", truncation=True, max_length=seq_length).to(device)


        # For CUDA, use built-in memory tracking.  For others, report 0.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            model.generate(**inputs, max_new_tokens=max_tokens)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
        else:
            peak_memory = 0.0  # No memory tracking for non-CUDA devices

        memory_results[seq_length] = peak_memory
        print(f"Sequence Length: {seq_length}, Peak Memory: {peak_memory:.2f} MB")

    return memory_results