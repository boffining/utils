from transformers import AutoTokenizer, AutoModelForCausalLM
import utils
import metrics
import os

if __name__ == "__main__":
    # Check if the target_model directory exists
    if os.path.isdir("/app/target_model"):
        model_name = "/app/target_model"
        print(f"Loading model from local directory: {model_name}")
    else:
        model_name = "gpt2" # Fallback if target_model is not provided.
        print(f"Loading default model: {model_name} (target_model not mounted)")

    prompt = "The quick brown fox jumps over the lazy"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)


    hardware_info = utils.get_hardware_info()
    print(f"Hardware Info: {hardware_info}")

    outputs, latency, throughput, token_throughput = metrics.evaluate_latency_throughput(model, tokenizer, prompt)
    # ... (rest of your evaluation code, using the 'metrics' functions) ...
    if hardware_info["device"] == "cuda":
        power_consumed, energy_per_token = metrics.evaluate_power_efficiency(model, tokenizer, prompt)
    precision_same = metrics.compare_precision_accuracy(model_name, prompt)
    memory_usage = metrics.memory_by_sequence_length(model, tokenizer, prompt)

    print(f"Memory Usage by Sequence Length: {memory_usage}")
    print(f"Precision Comparison (fp32 vs fp16): {'Identical' if precision_same else 'Different'}")