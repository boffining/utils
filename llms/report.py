import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName
from utils import (
    evaluate_latency_throughput,
    evaluate_power_efficiency,
    compare_precision_accuracy,
    memory_by_sequence_length,
)
from metrics import (
    calculate_perplexity,
    calculate_f1_score,
    calculate_precision_recall,
    calculate_mean_reciprocal_rank,
    calculate_mean_average_precision,
)


def get_hardware_info() -> dict:
    """
    Gather hardware details about the system used for evaluation.

    Returns:
        dict: A dictionary containing the number of GPUs, GPU types, and whether CUDA is available.
    """
    nvmlInit()
    gpu_count = nvmlDeviceGetCount()
    gpu_info = [nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)).decode() for i in range(gpu_count)]
    return {
        "num_gpus": gpu_count,
        "gpu_types": gpu_info,
        "cuda_available": torch.cuda.is_available(),
    }


def evaluate_model(
    model_name: str,
    evaluate_latency: bool = True,
    evaluate_power: bool = True,
    evaluate_precision: bool = True,
    evaluate_memory: bool = True,
    evaluate_metrics: bool = True,
    quantized: bool = False,
    prompt: str = "What is the impact of climate change?",
    batch_size: int = 4,
    max_tokens: int = 50,
    sequence_lengths: List[int] = [128, 256, 512, 1024],
    relevance_scores: List[List[int]] = [[1, 0, 1, 1, 0]],
    probabilities: List[float] = [0.2, 0.3, 0.1, 0.4],
    ranks: List[int] = [1, 3, 2, 0],
    precision: float = 0.8,
    recall: float = 0.75,
    true_positive: int = 50,
    false_positive: int = 10,
    false_negative: int = 15,
) -> Dict[str, Any]:
    """
    Evaluate an LLM using specified metrics and utilities.

    Args:
        model_name (str): The Hugging Face model name or path.
        evaluate_latency (bool): Whether to evaluate latency and throughput.
        evaluate_power (bool): Whether to evaluate power efficiency.
        evaluate_precision (bool): Whether to compare precision between fp32 and fp16.
        evaluate_memory (bool): Whether to measure memory usage for varying sequence lengths.
        evaluate_metrics (bool): Whether to compute various metrics like perplexity, F1 score, etc.
        quantized (bool): Whether the model is quantized or not.
        prompt (str): The input prompt for the model.
        batch_size (int): Batch size for evaluation.
        max_tokens (int): Maximum tokens to generate.
        sequence_lengths (List[int]): List of sequence lengths for memory evaluation.
        relevance_scores (List[List[int]]): Relevance scores for MAP calculation.
        probabilities (List[float]): Probabilities for perplexity calculation.
        ranks (List[int]): Ranks for MRR calculation.
        precision (float): Precision score for F1 calculation.
        recall (float): Recall score for F1 calculation.
        true_positive (int): True positives for precision/recall calculation.
        false_positive (int): False positives for precision/recall calculation.
        false_negative (int): False negatives for precision/recall calculation.

    Returns:
        Dict[str, Any]: A detailed report of all evaluations performed.

    Example:
        report = evaluate_model("meta-llama/Llama-2-7b-hf", evaluate_latency=True, evaluate_metrics=True)
        print(report)
    """
    report = {"model_name": model_name, "evaluation_results": {}, "conditions": {}}

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    # Hardware information
    hardware_info = get_hardware_info()
    model_parameters = sum(p.numel() for p in model.parameters())
    report["conditions"] = {
        "prompt": prompt,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "sequence_lengths": sequence_lengths,
        "num_gpus": hardware_info["num_gpus"],
        "gpu_types": hardware_info["gpu_types"],
        "cuda_available": hardware_info["cuda_available"],
        "model_parameters": model_parameters,
        "quantized": quantized,
    }

    # Latency and throughput evaluation
    if evaluate_latency:
        print("Evaluating latency and throughput...")
        outputs, latency, throughput, token_throughput = evaluate_latency_throughput(
            model, tokenizer, prompt, max_tokens, batch_size
        )
        report["evaluation_results"]["latency_throughput"] = {
            "latency": latency,
            "throughput": throughput,
            "token_throughput": token_throughput,
        }

    # Power efficiency evaluation
    if evaluate_power:
        print("Evaluating power efficiency...")
        power_consumed, energy_per_token = evaluate_power_efficiency(
            model, tokenizer, prompt, max_tokens, batch_size
        )
        report["evaluation_results"]["power_efficiency"] = {
            "power_consumed": power_consumed,
            "energy_per_token": energy_per_token,
        }

    # Precision comparison
    if evaluate_precision:
        print("Comparing precision...")
        precision_match = compare_precision_accuracy(model_name, prompt, max_tokens)
        report["evaluation_results"]["precision_comparison"] = {"precision_match": precision_match}

    # Memory evaluation
    if evaluate_memory:
        print("Evaluating memory usage...")
        memory_results = {}
        for length in sequence_lengths:
            memory_usage = memory_by_sequence_length(model, tokenizer, prompt, max_tokens, length)
            memory_results[f"sequence_length_{length}"] = memory_usage
        report["evaluation_results"]["memory_usage"] = memory_results

    # Metrics evaluation
    if evaluate_metrics:
        print("Evaluating metrics...")
        metrics_results = {
            "perplexity": calculate_perplexity(probabilities),
            "f1_score": calculate_f1_score(precision, recall),
            "precision_recall": calculate_precision_recall(true_positive, false_positive, false_negative),
            "mrr": calculate_mean_reciprocal_rank(ranks),
            "map": calculate_mean_average_precision(relevance_scores),
        }
        report["evaluation_results"]["metrics"] = metrics_results

    return report


# Example usage
if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    evaluation_report = evaluate_model(
        model_name,
        evaluate_latency=True,
        evaluate_power=True,
        evaluate_precision=True,
        evaluate_memory=True,
        evaluate_metrics=True,
        quantized=False,
    )
    print("\nEvaluation Report:")
    print(evaluation_report)
