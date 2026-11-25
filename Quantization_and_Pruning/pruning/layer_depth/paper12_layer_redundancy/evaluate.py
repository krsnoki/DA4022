"""
evaluate.py — Paper #12: Rethinking Layer Redundancy

Downstream evaluation functions:
  - Perplexity on WikiText-2    (for perplexity-pruned models)
  - Zero-shot accuracy on HellaSwag and BoolQ   (for TLM-pruned models)

These are EVALUATION functions — never used for calibration/search.

Reference: Table 1 in the paper (zero-shot accuracy on 6 reasoning tasks).
"""

import torch
import random
from torch.nn import CrossEntropyLoss
from calibration_objectives import score_completion


# ---------------------------------------------------------------------------
# Perplexity on WikiText-2
# ---------------------------------------------------------------------------

def evaluate_perplexity_wikitext2(model, tokenizer, stride: int = 512,
                                   max_length: int = 1024,
                                   device: str = "cuda") -> float:
    """
    Compute perplexity on WikiText-2 test split using a sliding-window approach.

    This is the standard evaluation protocol for language model perplexity.
    Uses overlapping windows of size max_length with step size stride to
    avoid boundary effects.

    Args:
        model:      HuggingFace CausalLM (eval mode, on device).
        tokenizer:  Matching tokenizer.
        stride:     Sliding window step size.
        max_length: Context window size.
        device:     'cuda' or 'cpu'.

    Returns:
        Perplexity (float). Lower is better.
    """
    from datasets import load_dataset

    model.eval()
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text    = "\n\n".join(dataset["text"])

    encodings  = tokenizer(text, return_tensors="pt")
    input_ids  = encodings["input_ids"].to(device)
    seq_len    = input_ids.size(1)

    total_loss   = 0.0
    total_tokens = 0
    loss_fn      = CrossEntropyLoss(reduction="sum")

    for begin_idx in range(0, seq_len, stride):
        end_idx    = min(begin_idx + max_length, seq_len)
        target_len = end_idx - (begin_idx + stride - stride)   # tokens actually predicted
        chunk      = input_ids[:, begin_idx:end_idx]

        # Only compute loss on the last target_len tokens of each chunk
        # to avoid double-counting overlapping context
        labels            = chunk.clone()
        labels[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(chunk, labels=labels)
            loss    = outputs.loss

        n_tokens      = (labels != -100).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

        if end_idx == seq_len:
            break

    avg_nll = total_loss / max(total_tokens, 1)
    return torch.exp(torch.tensor(avg_nll)).item()


# ---------------------------------------------------------------------------
# Zero-shot accuracy on HellaSwag
# ---------------------------------------------------------------------------

def evaluate_hellaswag(model, tokenizer, n_samples: int = None,
                        device: str = "cuda", seed: int = 42) -> float:
    """
    Zero-shot accuracy on HellaSwag validation split.

    For each example, scores all 4 candidate endings using log P(ending | context)
    and predicts the highest-scoring one.

    Args:
        model:     HuggingFace CausalLM.
        tokenizer: Matching tokenizer.
        n_samples: If None, evaluate on the full validation set (~10k examples).
        device:    'cuda' or 'cpu'.
        seed:      Random seed for sampling (when n_samples is set).

    Returns:
        Accuracy (float, 0–100).
    """
    from datasets import load_dataset

    random.seed(seed)
    model.eval()

    dataset = load_dataset("hellaswag", split="validation")
    if n_samples is not None:
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        dataset = dataset.select(indices)

    correct = 0
    total   = 0

    for example in dataset:
        context  = example.get("ctx", example.get("ctx_a", "") + " " + example.get("ctx_b", ""))
        endings  = example["endings"]
        label    = int(example["label"])

        scores = [
            score_completion(model, tokenizer, context, e, device)
            for e in endings
        ]
        pred = scores.index(max(scores))
        if pred == label:
            correct += 1
        total += 1

    return 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Zero-shot accuracy on BoolQ
# ---------------------------------------------------------------------------

def evaluate_boolq(model, tokenizer, n_samples: int = None,
                   device: str = "cuda", seed: int = 42) -> float:
    """
    Zero-shot accuracy on BoolQ validation split.

    Scores "Yes" and "No" completions given the passage + question,
    predicts the higher-scoring one.

    Args:
        model:     HuggingFace CausalLM.
        tokenizer: Matching tokenizer.
        n_samples: If None, evaluate on the full validation set (~3k examples).
        device:    'cuda' or 'cpu'.
        seed:      Random seed.

    Returns:
        Accuracy (float, 0–100).
    """
    from datasets import load_dataset

    random.seed(seed)
    model.eval()

    dataset = load_dataset("boolq", split="validation")
    if n_samples is not None:
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        dataset = dataset.select(indices)

    correct = 0
    total   = 0

    for example in dataset:
        passage  = example["passage"]
        question = example["question"]
        label    = example["answer"]   # True or False

        context  = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
        yes_score = score_completion(model, tokenizer, context, " Yes", device)
        no_score  = score_completion(model, tokenizer, context, " No",  device)

        pred = yes_score > no_score   # True if "Yes" scores higher
        if pred == label:
            correct += 1
        total += 1

    return 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Combined evaluation report
# ---------------------------------------------------------------------------

def evaluate_all(model, tokenizer, device: str = "cuda",
                 n_samples: int = None, seed: int = 42) -> dict:
    """
    Run all three evaluation metrics and return a results dict.

    Args:
        model:     HuggingFace CausalLM (pruned or baseline).
        tokenizer: Matching tokenizer.
        device:    'cuda' or 'cpu'.
        n_samples: Subsample size for HellaSwag and BoolQ (None = full set).
        seed:      Random seed.

    Returns:
        dict with keys: perplexity_wikitext2, hellaswag_acc, boolq_acc
    """
    print("  Evaluating: WikiText-2 perplexity...")
    ppl = evaluate_perplexity_wikitext2(model, tokenizer, device=device)

    print("  Evaluating: HellaSwag zero-shot accuracy...")
    hella = evaluate_hellaswag(model, tokenizer, n_samples=n_samples,
                                device=device, seed=seed)

    print("  Evaluating: BoolQ zero-shot accuracy...")
    boolq = evaluate_boolq(model, tokenizer, n_samples=n_samples,
                            device=device, seed=seed)

    return {
        "perplexity_wikitext2": round(ppl,   4),
        "hellaswag_acc":        round(hella, 2),
        "boolq_acc":            round(boolq, 2),
    }
