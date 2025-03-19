from transformers import AutoTokenizer, AutoModelForCausalLM
import utils
import metric_tracker
import os
# from huggingface_hub import login
import logging


LOG_FILE = "my_llm_log.txt"
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
#     level=logging.DEBUG
)
# logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    # --- Hugging Face Login (BEFORE model loading) ---
#     hf_token = os.environ.get("HF_TOKEN")
#     if hf_token:
#         try:
#             login(token=hf_token)
#             print("Successfully logged in to Hugging Face Hub.")
#         except Exception as e:
#             print(f"Hugging Face login failed: {e}")
#             #  Crucially, DO NOT exit here.  We might still
#             #  be able to load a cached or public model.
#     else:
#         print("HF_TOKEN environment variable not set.  Attempting to load without authentication.")
    # --- End of Hugging Face Login ---

    
    # Example query function (replace with your actual LLM query function)
    @metric_tracker.evaluate_llm(log_file=LOG_FILE)  # Apply the decorator
    def generate_text(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 50) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(utils.get_device())
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Check for target_model first
    if os.path.isdir("/app/target_model"):
        model_name = "/app/target_model"
        print(f"Loading model from local directory: {model_name}")
    else:
        model_name = "gpt2"  # Fallback if target_model is not provided.
        print(f"Loading default model: {model_name} (target_model not mounted)")

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(utils.get_device())
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1) #Exit here because if the model does not load we don't want to continue

    prompt = "Once upon a time"

    # Call the decorated function.
    generated_text, execution_time = generate_text(model, tokenizer, prompt)
    print(f"Generated Text:\n{generated_text}")
    print(f"Execution Time for generation: {execution_time}")




#     try:
#         model_name = "gpt2"
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#         print(f"Successfully loaded model: {model_name}")

#         prompt = "Once upon a time"
#         inputs = tokenizer(prompt, return_tensors="pt")
#         outputs = model.generate(**inputs, max_new_tokens=50)
#         generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(f"Generated text: {generated_text}")

#     except Exception as e:
#         print(f"An error occurred: {e}")

#     print("Finished.")