import time
import torch
from typing import Callable, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
import utils
import metrics
import functools
import logging
import sys


def log_environment_and_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, logger: logging.Logger):
    """Logs environment and model information."""
    hardware_info = utils.get_hardware_info()

    logger.info("Environment and Model Information:")
    logger.info(f"  Device: {hardware_info['device']}")
    logger.info(f"  Number of GPUs: {hardware_info['num_gpus']}")
    logger.info(f"  GPU Types: {', '.join(hardware_info['gpu_types'])}")
    logger.info(f"  Model Name: {getattr(model.config, 'name_or_path', 'N/A')}")
    logger.info(f"  Tokenizer: {type(tokenizer).__name__}")

    # Attempt to get more detailed model configuration.  Handle potential errors.
    try:
        logger.info(f"  Model Architecture: {model.config.architectures[0] if model.config.architectures else 'N/A'}")
        logger.info(f"  Hidden Size: {model.config.hidden_size}")
        logger.info(f"  Number of Layers: {model.config.num_hidden_layers}")
        logger.info(f"  Number of Attention Heads: {model.config.num_attention_heads}")
        logger.info(f"  Temperature: {getattr(model.config, 'temperature', 'N/A')}")  # Use getattr for optional params
        logger.info(f"  Top-p: {getattr(model.config, 'top_p', 'N/A')}")
        logger.info(f"  Top-k: {getattr(model.config, 'top_k', 'N/A')}")
        logger.info(f"  Repetition Penalty: {getattr(model.config, 'repetition_penalty', 'N/A')}")

    except Exception as e:
        logger.warning(f"Error retrieving detailed model config: {e}")
        logger.info("  (Some model details could not be accessed)")


def evaluate_llm(log_file) -> Callable:
    """
    Decorator to evaluate LLM performance and log metrics.

    Args:
        log_file (str): Path to the log file. Defaults to "llm_evaluation.log".

    Returns:
        Callable: The decorated function.
    """

    def decorator(query_function: Callable) -> Callable:
        @functools.wraps(query_function)  # Preserve original function metadata
        def wrapper(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, *args, **kwargs) -> Any:

            # Configure logging
#             logging.basicConfig(
#                 filename=log_file,
#                 filemode='a',  # Append to the log file
#                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                 level=logging.INFO
#             )
            logger = logging.getLogger("LLMEvaluator")
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    
            logger.info("=" * 30)

            log_environment_and_model_info(model, tokenizer, logger)

            # --- Performance Evaluation ---
            logger.info("Performance Metrics:")

            try:
                _, latency, throughput, token_throughput = metrics.evaluate_latency_throughput(
                    model, tokenizer, prompt, *args, **kwargs
                )
                logger.info(f"  Latency: {latency:.4f} seconds")
                logger.info(f"  Throughput: {throughput:.4f} responses/second")
                logger.info(f"  Token Throughput: {token_throughput:.4f} tokens/second")
            except Exception as e:
                logger.error(f"Error during latency/throughput evaluation: {e}")


            try:
                if utils.get_device().type == "cuda" and utils.NVML_AVAILABLE:
                    power_consumed, energy_per_token = metrics.evaluate_power_efficiency(
                        model, tokenizer, prompt, *args, **kwargs
                    )
                    logger.info(f"  Power Consumption: {power_consumed:.4f} W")
                    logger.info(f"  Energy per Token: {energy_per_token:.6f} W/token")
                else:
                    logger.info("  Power efficiency evaluation skipped (not CUDA or no NVML).")

            except Exception as e:
                logger.error(f"Error during power efficiency evaluation: {e}")
            logger.info("=" * 30)
             # --- Execute the Original Function ---
            start_time = time.time()
            result = query_function(model, tokenizer, prompt, *args, **kwargs)
            end_time = time.time()
            execution_time = end_time-start_time

            return result, execution_time  # Return the result and execution time

        return wrapper

    return decorator