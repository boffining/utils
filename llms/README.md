# Metrics for Evaluating Language Models

This repository provides implementations for key metrics to evaluate Large Language Models (LLMs). Each metric includes documentation, usage examples, and references to sources or research papers.

---

## **Metrics Overview**

### 1. Perplexity
- **Description**: Measures how well a language model predicts a sample.
- **Formula**: \( \text{Perplexity} = e^{-\frac{1}{N} \sum_{i=1}^N \log(p_i)} \)
- **Use Case**: Evaluate fluency and predictability of generated text.
- **Source**: [Hugging Face Perplexity Metric](https://huggingface.co/docs/evaluate/metrics/perplexity)
- **Example Image**:
  ![Perplexity Visualization](https://tse3.mm.bing.net/th?id=OIP.r0qms0mViLFBO8UaMYBPhwHaFv&pid=Api)

---

### 2. F1 Score
- **Description**: Balances precision and recall.
- **Formula**: \( F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \)
- **Use Case**: Evaluate accuracy for classification tasks.
- **Source**: [Hugging Face F1 Metric](https://huggingface.co/docs/evaluate/metrics/f1)
- **Example Image**:
  ![F1 Score Visualization](https://tse1.mm.bing.net/th?id=OIP.Wcm44-sIynUXsvr-u_jtDAHaC1&pid=Api)

---

### 3. Precision and Recall
- **Description**:
  - **Precision**: Proportion of true positives among all positive predictions.
  - **Recall**: Proportion of true positives identified out of all actual positives.
- **Use Case**: Crucial for classification and information retrieval.
- **Source**: [Hugging Face Precision Metric](https://huggingface.co/docs/evaluate/metrics/precision)
- **Example Image**:
  ![Precision and Recall](https://tse4.mm.bing.net/th?id=OIP.rvOkfUasq6MpD9kxbr2HOAHaER&pid=Api)

---

### 4. Mean Reciprocal Rank (MRR)
- **Description**: Evaluates ranking quality by measuring reciprocal rank of the first relevant result.
- **Formula**: \( MRR = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q} \)
- **Source**: [Hugging Face MRR Metric](https://huggingface.co/docs/evaluate/metrics/mrr)
- **Example Image**:
  ![MRR Visualization](https://tse1.mm.bing.net/th?id=OIP.tDedgOY5yiS5dRtMvoQXBgHaEK&pid=Api)

---

### 5. Mean Average Precision (MAP)
- **Description**: Evaluates precision at all relevant documents for a query.
- **Formula**: Average precision per query, averaged across all queries.
- **Source**: [Hugging Face MAP Metric](https://huggingface.co/docs/evaluate/metrics/map)
- **Example Image**:
  ![MAP Visualization](https://tse3.mm.bing.net/th?id=OIP.HD9_4YqQHan0lJ-WfnIxfgHaEL&pid=Api)

---

## **How to Use**

1. Import the desired metric function from `metrics.py`.
2. Pass the required inputs (e.g., probabilities, ranks, relevance scores).
3. Review results and interpret the metric as described.

Example:
```python
from metrics import calculate_perplexity

probabilities = [0.2, 0.4, 0.1, 0.3]
perplexity = calculate_perplexity(probabilities)
print(f"Perplexity: {perplexity}")
```



# LLM Evaluation Tool

This tool evaluates the performance of Large Language Models (LLMs) within a Docker container. It measures latency, throughput, power efficiency (on CUDA devices), memory usage, and precision, and provides detailed logging.

## Prerequisites

*   Docker installed and running.
*   (Optional) An LLM model in a local directory for offline use. This model must be in the Hugging Face Transformers format (i.e., have a `config.json`, `pytorch_model.bin` or similar, and tokenizer files).

## Files

*   `Dockerfile`: Builds the Docker image.
*   `requirements.txt`: Lists Python dependencies.
*   `utils.py`: Utility functions (device detection, hardware info).
*   `metrics.py`: Core metric calculation functions (latency, throughput, power, etc.).
*   `metric_tracker.py`: Decorator for automatic metric tracking and logging.
*   `test.py`: Unit tests (using `pytest`).
*   `setup.sh`: Builds the Docker image and runs initial tests.
*   `shutdown.sh`: (Optional) Cleanup script.
*   `main.py`: Example script for running evaluations.

## Setup

1.  **Directory Structure:** Place all the provided files (`Dockerfile`, `requirements.txt`, `utils.py`, `metrics.py`, `metric_tracker.py`, `test.py`, `setup.sh`, `shutdown.sh`, `main.py`) in the same directory.

2.  **(Optional) Local Model:** If you want to evaluate a *local* model, create a subdirectory (e.g., `target_model`) and place your model files *inside* that subdirectory.  The directory structure should look like this:

    ```
    your_project_directory/
    ├── Dockerfile
    ├── requirements.txt
    ├── utils.py
    ├── metrics.py
    ├── metric_tracker.py
    ├── test.py
    ├── setup.sh
    ├── shutdown.sh
    ├── main.py
    └── target_model/       <-- Your local model directory
        ├── config.json
        ├── pytorch_model.bin  (or similar model files)
        ├── tokenizer.json
        └── ... other tokenizer files ...
    ```

    You can create the `target_model` directory and download a model (e.g., `sshleifer/tiny-gpt2`) like this (replace with your desired model):

    ```bash
    mkdir my_model
    transformers-cli download sshleifer/tiny-gpt2 --local-dir ./my_model --local-dir-use-symlinks False
    ```
    The `--local-dir-use-symlinks False` flag is important for Docker volume mounting.

3.  **Build and Test:** Run the `setup.sh` script:

    ```bash
    chmod +x setup.sh  # Make the script executable (if needed)
    ./setup.sh
    ```

    This script builds the Docker image (tagged as `llm-eval-test`) and then runs the unit tests *inside* a container to ensure that the base environment (Python, libraries) is set up correctly.  If any tests fail, you'll see error messages in the terminal.

## Usage

### Running with a Local Model (Offline)

Use the `-v` option with `docker run` to mount your local model directory into the container at `/app/target_model`.  This makes the model available to the scripts inside the container.

```bash
docker run -v /path/to/your/target_model:/app/target_model llm-eval-test python main.py
