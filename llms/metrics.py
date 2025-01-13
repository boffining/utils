"""
metrics.py

This module contains implementations and utilities to evaluate language models (LLMs) based on various
metrics, including BLEU, ROUGE, METEOR, BERTScore, and others.

The metrics are demonstrated with the Hugging Face model "Mistral" as an example.

Requirements:
- transformers
- torch
- datasets
- bert_score
- nltk

Install dependencies:
    pip install transformers torch datasets bert-score nltk

"""

from typing import List, Dict, Tuple
#from datasets import load_metric
from evaluate import load as load_metric
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
import numpy as np
import math

# Example Hugging Face model for evaluation
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

def calculate_bleu(predictions: List[str], references: List[List[str]]) -> Dict:
    """
    Calculate BLEU score for the model's predictions.

    Args:
        predictions (List[str]): A list of predicted texts from the model.
        references (List[List[str]]): A list of reference texts (ground truth).

    Returns:
        Dict: BLEU score and additional metrics.

    Resource:
        https://github.com/huggingface/evaluate
    """
    bleu_metric = load_metric("bleu")
    bleu_metric.add_batch(predictions=predictions, references=references)
    result = bleu_metric.compute()
    return result

def calculate_rouge(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate ROUGE score for the model's predictions.

    Args:
        predictions (List[str]): A list of predicted texts from the model.
        references (List[str]): A list of reference texts (ground truth).

    Returns:
        Dict: ROUGE scores.

    Resource:
        https://github.com/huggingface/evaluate
    """
    rouge_metric = load_metric("rouge")
    rouge_metric.add_batch(predictions=predictions, references=references)
    result = rouge_metric.compute()
    return result

def calculate_meteor(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate METEOR score for the model's predictions.

    Args:
        predictions (List[str]): A list of predicted texts from the model.
        references (List[str]): A list of reference texts (ground truth).

    Returns:
        Dict: METEOR score.

    Resource:
        https://github.com/huggingface/evaluate
    """
    meteor_metric = load_metric("meteor")
    meteor_metric.add_batch(predictions=predictions, references=references)
    result = meteor_metric.compute()
    return result

def calculate_bert_score(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate BERTScore for the model's predictions.

    Args:
        predictions (List[str]): A list of predicted texts from the model.
        references (List[str]): A list of reference texts (ground truth).

    Returns:
        Dict: Precision, Recall, and F1 scores.

    Resource:
        https://github.com/Tiiiger/bert_score
    """
    P, R, F1 = score(predictions, references, lang="en", verbose=True)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

def calculate_ragas_score(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate RAGAS (Retrieval-Augmented Generation Answer Score).

    Args:
        predictions (List[str]): A list of predicted texts from the model.
        references (List[str]): A list of reference texts (ground truth).

    Returns:
        Dict: RAGAS score.

    Resource:
        https://github.com/explodinggradients/ragas
    """
    from ragas import evaluate
    ragas_result = evaluate(predictions, references)
    return {"ragas_score": ragas_result["score"]}

def calculate_helm_score(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate HELM (Holistic Evaluation of Language Models) metrics.

    Args:
        predictions (List[str]): A list of predicted texts from the model.
        references (List[str]): A list of reference texts (ground truth).

    Returns:
        Dict: HELM score.

    Resource:
        https://crfm.stanford.edu/helm/latest/
    """
    # Placeholder for actual HELM evaluation framework integration
    helm_score = np.mean([len(pred) / max(len(ref), 1) for pred, ref in zip(predictions, references)])
    return {"helm_score": helm_score}

def calculate_gpt_score(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate GPT-Score for text similarity.

    Args:
        predictions (List[str]): A list of predicted texts from the model.
        references (List[str]): A list of reference texts (ground truth).

    Returns:
        Dict: GPT-Score values.

    Resource:
        https://github.com/IntelLabs/gpt-score
    """
    from gpt_score import GPTScorer
    scorer = GPTScorer()
    scores = scorer.score(predictions, references)
    return {"gpt_score": np.mean(scores)}

def calculate_forgetting_rate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: List[str]) -> float:
    """
    Calculate Forgetting Rate of the model over repeated evaluations.

    Args:
        model (AutoModelForCausalLM): The language model to test.
        tokenizer (AutoTokenizer): Tokenizer associated with the model.
        prompts (List[str]): List of input prompts.

    Returns:
        float: Forgetting rate as a percentage.

    Resource:
        https://arxiv.org/abs/2205.12647
    """
    baseline_results = []
    repeated_results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        baseline_output = model.generate(**inputs)
        repeated_output = model.generate(**inputs)

        baseline_results.append(tokenizer.decode(baseline_output[0], skip_special_tokens=True))
        repeated_results.append(tokenizer.decode(repeated_output[0], skip_special_tokens=True))

    differences = [1 if b != r else 0 for b, r in zip(baseline_results, repeated_results)]
    forgetting_rate = sum(differences) / len(prompts) * 100
    return forgetting_rate

def calculate_brevity_score(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate Brevity Score to evaluate concise text generation.

    Args:
        predictions (List[str]): A list of predicted texts from the model.
        references (List[str]): A list of reference texts (ground truth).

    Returns:
        Dict: Brevity score.

    Resource:
        https://arxiv.org/pdf/1904.09675.pdf
    """
    brevity_ratios = [len(pred.split()) / max(len(ref.split()), 1) for pred, ref in zip(predictions, references)]
    brevity_score = np.mean([min(1.0, ratio) for ratio in brevity_ratios])
    return {"brevity_score": brevity_score}

def calculate_perplexity(probabilities: List[float]) -> float:
    """
    Calculate the perplexity of a model's output.

    Args:
        probabilities (List[float]): A list of probabilities for each token in the sequence.

    Returns:
        float: The perplexity score.

    Reference:
        - https://huggingface.co/docs/evaluate/metrics/perplexity
    """
    cross_entropy = -np.mean(np.log(probabilities))
    perplexity = np.exp(cross_entropy)
    return perplexity


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate the F1 score given precision and recall.

    Args:
        precision (float): Precision of the predictions.
        recall (float): Recall of the predictions.

    Returns:
        float: The F1 score.

    Reference:
        - https://huggingface.co/docs/evaluate/metrics/f1
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_precision_recall(true_positive: int, false_positive: int, false_negative: int) -> Dict[str, float]:
    """
    Calculate precision and recall.

    Args:
        true_positive (int): Number of true positive cases.
        false_positive (int): Number of false positive cases.
        false_negative (int): Number of false negative cases.

    Returns:
        Dict[str, float]: A dictionary containing precision and recall scores.

    Reference:
        - https://huggingface.co/docs/evaluate/metrics/precision
    """
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    return {"precision": precision, "recall": recall}


def calculate_mean_reciprocal_rank(ranks: List[int]) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR).

    Args:
        ranks (List[int]): A list of ranks for the first relevant result in each query.

    Returns:
        float: The MRR score.

    Reference:
        - https://huggingface.co/docs/evaluate/metrics/mrr
    """
    reciprocal_ranks = [1 / rank if rank > 0 else 0 for rank in ranks]
    return np.mean(reciprocal_ranks)


def calculate_mean_average_precision(relevance_scores: List[List[int]]) -> float:
    """
    Calculate Mean Average Precision (MAP).

    Args:
        relevance_scores (List[List[int]]): A list of binary relevance scores for each query's retrieved documents.

    Returns:
        float: The MAP score.

    Reference:
        - https://huggingface.co/docs/evaluate/metrics/map
    """
    average_precisions = []
    for scores in relevance_scores:
        precision_at_k = [
            sum(scores[:k + 1]) / (k + 1) for k in range(len(scores)) if scores[k] == 1
        ]
        if precision_at_k:
            average_precisions.append(np.mean(precision_at_k))
    return np.mean(average_precisions) if average_precisions else 0.0

"""
Usage Example with Hugging Face LLM Mistral:

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

    # Sample inputs
    predictions = ["Climate change is a global challenge that requires..."]
    references = [["Climate change is a pressing issue affecting..."]]

    # BLEU
    print("BLEU Score:", calculate_bleu(predictions, references))

    # ROUGE
    print("ROUGE Score:", calculate_rouge(predictions, [r[0] for r in references]))

    # METEOR
    print("METEOR Score:", calculate_meteor(predictions, [r[0] for r in references]))

    # BERTScore
    print("BERTScore:", calculate_bert_score(predictions, [r[0] for r in references]))

    # RAGAS
    print("RAGAS Score:", calculate_ragas_score(predictions, [r[0] for r in references]))

    # HELM
    print("HELM Score:", calculate_helm_score(predictions, [r[0] for r in references]))

    # GPT-Score
    print("GPT-Score:", calculate_gpt_score(predictions, [r[0] for r in references]))

    # Forgetting Rate
    prompts = ["What is climate change?", "Explain photosynthesis."]
    print("Forgetting Rate:", calculate_forgetting_rate(model, tokenizer, prompts))

    # Brevity Score
    print("Brevity Score:", calculate_brevity_score(predictions, [r[0] for r in references]))
"""
