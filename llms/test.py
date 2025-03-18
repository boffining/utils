import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import get_device, get_hardware_info
from metrics import (evaluate_latency_throughput, evaluate_power_efficiency,
                   compare_precision_accuracy, memory_by_sequence_length)


# Fixture for a small, fast model and tokenizer
@pytest.fixture(scope="module")
def small_model_and_tokenizer():
    # Use a tiny, fast model for testing
    model_name = "sshleifer/tiny-gpt2"  # Very small GPT-2 model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def test_get_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda", "mps")

def test_get_hardware_info():
    hardware_info = get_hardware_info()
    assert isinstance(hardware_info, dict)
    assert "num_gpus" in hardware_info
    assert "gpu_types" in hardware_info
    assert "device" in hardware_info

def test_evaluate_latency_throughput(small_model_and_tokenizer):
    model, tokenizer = small_model_and_tokenizer
    prompt = "test"
    outputs, latency, throughput, token_throughput = evaluate_latency_throughput(
        model, tokenizer, prompt, max_tokens=10, batch_size=1
    )
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(latency, float) and latency > 0
    assert isinstance(throughput, float) and throughput > 0
    assert isinstance(token_throughput, float) and token_throughput > 0

def test_evaluate_power_efficiency(small_model_and_tokenizer):
    model, tokenizer = small_model_and_tokenizer
    prompt = "test"
    # Mock pynvml if not available to avoid errors during testing.
    try:
        import pynvml  # Try importing to see if it's available
        power_consumed, energy_per_token = evaluate_power_efficiency(
            model, tokenizer, prompt, max_tokens=10, batch_size=1
        )
        assert isinstance(power_consumed, float)
        assert isinstance(energy_per_token, float)
    except ImportError:
        # If pynvml import fails, it means it is not installed,
        # and the test should pass, returning 0.0, 0.0
        power_consumed, energy_per_token = evaluate_power_efficiency(model,tokenizer,prompt)
        assert power_consumed == 0.0
        assert energy_per_token == 0.0


def test_compare_precision_accuracy(small_model_and_tokenizer):
    _, tokenizer = small_model_and_tokenizer  # We only need the tokenizer for getting model name
    prompt = "test"
    # Use the tokenizer to get the *name* of the model. We don't want to
    # load the large model within the fixture.
    model_name = tokenizer.name_or_path
    is_identical = compare_precision_accuracy(model_name, prompt, max_tokens=10)
    assert isinstance(is_identical, bool)

def test_memory_by_sequence_length(small_model_and_tokenizer):
    model, tokenizer = small_model_and_tokenizer
    prompt = "test"
    memory_results = memory_by_sequence_length(
        model, tokenizer, prompt, max_tokens=10, max_length=128
    )
    assert isinstance(memory_results, dict)
    assert 128 in memory_results  # Check for at least one tested length
    assert all(isinstance(value, float) for value in memory_results.values())

    # Check that if it's not CUDA, all values are 0.
    if get_device().type != "cuda":
        assert all(value == 0.0 for value in memory_results.values())
        
def test_evaluate_llm_decorator(small_model_and_tokenizer, tmp_path):
    model, tokenizer = small_model_and_tokenizer
    prompt = "Hello, world!"
    log_file = tmp_path / "test_log.txt"

    @metric_tracker.evaluate_llm(log_file=str(log_file))
    def dummy_query_function(model, tokenizer, prompt):
        return "Mocked result"

    result, execution_time = dummy_query_function(model, tokenizer, prompt)
    assert result == "Mocked result"
    assert isinstance(execution_time, float)
    assert execution_time > 0

    # Check if the log file was created and contains expected content
    assert log_file.exists()
    with open(log_file, "r") as f:
        log_content = f.read()
    assert "Performance Metrics:" in log_content
    assert "Latency:" in log_content
    assert "Environment and Model Information:" in log_content

    # Check if the log file contains information correctly
    assert "Device:" in log_content
    assert "Model Name:" in log_content
    assert "tiny-gpt2" in log_content # Check specifically for the model used

def test_log_environment_and_model_info(small_model_and_tokenizer, caplog):
    model, tokenizer = small_model_and_tokenizer

    # Configure logging to capture log messages
    caplog.set_level(logging.INFO)

    # Create a logger instance
    logger = logging.getLogger("TestLogger")

    # Call the function with the model, tokenizer, and logger
    metric_tracker.log_environment_and_model_info(model, tokenizer, logger)

     # Check for expected log messages.  Use caplog.text to get all captured logs.
    assert "Environment and Model Information:" in caplog.text
    assert "Device:" in caplog.text
    assert "Model Name:" in caplog.text
    assert "tiny-gpt2" in caplog.text  # Check specifically for the model used
