"""
utils.py

This module contains utilities for evaluating the performance of large language models (LLMs) using
metrics such as latency, throughput, power consumption, memory usage, and precision comparison.
It is designed for use with Hugging Face's open-source LLMs, including models like Llama 2 7B.

Requirements:
- transformers
- torch
- pynvml

Install dependencies:
    pip install transformers torch pynvml

"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetMemoryInfo
from typing import Tuple, List

# Initialize NVIDIA Management Library
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)

# Helper functions
def track_power() -> float:
    """Track GPU power consumption in watts."""
    return nvmlDeviceGetPowerUsage(gpu_handle) / 1000

def track_memory() -> float:
    """Track GPU memory usage in GB."""
    mem_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    return mem_info.used / (1024 ** 3)

def count_model_parameters(model: torch.nn.Module) -> int:
    """Count the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

class LLMPerformanceTester:
    """
    A utility class for evaluating the performance of Hugging Face LLMs.

    Attributes:
        model_name (str): The Hugging Face model name to load.
        tokenizer: The tokenizer associated with the model.
        model: The LLM model loaded from Hugging Face.
        device (str): Device to run the model on ('cuda' or 'cpu').
        parameter_count (int): Number of parameters in the model.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parameter_count = count_model_parameters(self.model)
        print(f"Model Size: {self.parameter_count:,} parameters")

    def evaluate_latency_throughput(self, prompt: str, max_tokens: int = 50, batch_size: int = 1) -> Tuple[torch.Tensor, float, float, float]:
        """
        Evaluate latency, token throughput, and token processing rate.

        Args:
            prompt (str): Input prompt for the model.
            max_tokens (int): Maximum number of tokens to generate.
            batch_size (int): Number of prompts in a batch.

        Returns:
            Tuple[torch.Tensor, float, float, float]: Generated outputs, latency, throughput, and token throughput.
        """
        inputs = [prompt] * batch_size
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

        # Warm-up
        print("Warming up...")
        start_warmup = time.time()
        self.model.generate(**tokenized_inputs, max_new_tokens=max_tokens)
        end_warmup = time.time()
        warmup_time = end_warmup - start_warmup
        print(f"Warm-up Time: {warmup_time:.2f}s")

        # Measure latency and throughput
        print("Measuring latency and throughput...")
        start_time = time.time()
        outputs = self.model.generate(**tokenized_inputs, max_new_tokens=max_tokens)
        end_time = time.time()

        latency = end_time - start_time
        throughput = batch_size / latency
        total_tokens = max_tokens * batch_size
        token_throughput = total_tokens / latency
        print(f"Latency: {latency:.2f}s | Throughput: {throughput:.2f} responses/sec | Token Throughput: {token_throughput:.2f} tokens/sec")
        return outputs, latency, throughput, token_throughput

    def evaluate_power_efficiency(self, prompt: str, max_tokens: int = 50, batch_size: int = 1) -> Tuple[float, float]:
        """
        Evaluate power consumption and efficiency per token.

        Args:
            prompt (str): Input prompt for the model.
            max_tokens (int): Maximum number of tokens to generate.
            batch_size (int): Number of prompts in a batch.

        Returns:
            Tuple[float, float]: Total power consumed and energy per token.
        """
        inputs = [prompt] * batch_size
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

        # Warm-up
        print("Warming up...")
        self.model.generate(**tokenized_inputs, max_new_tokens=max_tokens)

        # Measure power and efficiency
        print("Measuring power efficiency...")
        power_start = track_power()
        start_time = time.time()

        self.model.generate(**tokenized_inputs, max_new_tokens=max_tokens)

        end_time = time.time()
        power_end = track_power()

        latency = end_time - start_time
        throughput = batch_size / latency
        power_consumed = (power_end - power_start) * latency
        total_tokens = max_tokens * batch_size
        energy_per_token = power_consumed / total_tokens if total_tokens > 0 else float('inf')

        print(f"Power Consumption: {power_consumed:.2f} W | Energy per Token: {energy_per_token:.4f} W/token")
        return power_consumed, energy_per_token

    def compare_precision_accuracy(self, prompt: str, max_tokens: int = 50) -> bool:
        """
        Compare outputs for fp32 and fp16 precision to test integrity.

        Args:
            prompt (str): Input prompt for the model.
            max_tokens (int): Maximum number of tokens to generate.

        Returns:
            bool: Whether the outputs match between fp32 and fp16 precisions.
        """
        # Full precision (fp32)
        model_fp32 = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_fp32 = model_fp32.generate(**inputs, max_new_tokens=max_tokens)
        text_fp32 = self.tokenizer.decode(output_fp32[0], skip_special_tokens=True)

        # Mixed precision (fp16)
        model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        output_fp16 = model_fp16.generate(**inputs, max_new_tokens=max_tokens)
        text_fp16 = self.tokenizer.decode(output_fp16[0], skip_special_tokens=True)

        print(f"Full Precision (fp32) Output:\n{text_fp32}")
        print(f"Mixed Precision (fp16) Output:\n{text_fp16}")

        similarity = text_fp32 == text_fp16
        print(f"Outputs Match: {similarity}")
        return similarity

    def memory_by_sequence_length(self, base_prompt: str, max_tokens: int = 50, max_length: int = 1024) -> None:
        """
        Evaluate memory usage as sequence length increases.

        Args:
            base_prompt (str): Base string to repeat for increasing sequence length.
            max_tokens (int): Maximum number of tokens to generate.
            max_length (int): Maximum sequence length to test.
        """
        print("Memory Usage by Sequence Length:")
        for seq_length in [128, 256, 512, max_length]:
            prompt = base_prompt * (seq_length // len(base_prompt))
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

            torch.cuda.reset_peak_memory_stats()
            self.model.generate(**inputs, max_new_tokens=max_tokens)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB

            print(f"Sequence Length: {seq_length} | Peak Memory Usage: {peak_memory:.2f} MB")

"""
Usage example:

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"  # Example model
    prompt = "Explain the impact of climate change on global agriculture."
    base_prompt = "Climate change affects agriculture in multiple ways. "

    tester = LLMPerformanceTester(model_name)

    # Latency and token throughput
    tester.evaluate_latency_throughput(prompt, max_tokens=50, batch_size=4)

    # Power efficiency and energy per token
    tester.evaluate_power_efficiency(prompt, max_tokens=50, batch_size=4)

    # Compare fp32 and fp16 outputs
    tester.compare_precision_accuracy(prompt, max_tokens=50)

    # Memory by sequence length
    tester.memory_by_sequence_length(base_prompt, max_tokens=50, max_length=1024)
"""
