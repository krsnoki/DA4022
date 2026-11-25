"""
calibration_objectives.py — Paper #12: Rethinking Layer Redundancy

Two calibration objectives used to score pruned models:
  1. Perplexity  — negative log-likelihood on C4 (task-agnostic)
  2. Task Likelihood Margin (TLM) — from Wee et al. 2025

Lower perplexity = better.
More negative TLM margin = better (model more confident on correct answer).

Reference:
  Wee et al. 2025 — Task Likelihood Margin loss for layer pruning calibration.
  Raffel et al. 2020 — C4 dataset.
"""

import torch
import random
from torch.nn import CrossEntropyLoss


# ---------------------------------------------------------------------------
# Objective 1: Perplexity on C4
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, dataset_name: str = "allenai/c4",
                       dataset_config: str = "en", split: str = "train",
                       n_samples: int = 128, max_length: int = 512,
                       batch_size: int = 4, device: str = "cuda",
                       seed: int = 42) -> float:
    """
    Compute perplexity (exp of avg NLL) on a text dataset.

    Uses teacher-forcing: the model predicts each token given all previous tokens.
    Calibration split is kept separate from evaluation split.

    Args:
        model:          HuggingFace CausalLM (already on device, eval mode).
        tokenizer:      Matching tokenizer.
        dataset_name:   HuggingFace dataset identifier (default: allenai/c4).
        dataset_config: Dataset config/subset (default: en).
        split:          Dataset split to sample from.
        n_samples:      Number of text examples to use.
        max_length:     Tokenisation truncation length.
        batch_size:     Inference batch size.
        device:         'cuda' or 'cpu'.
        seed:           Random seed for reproducible sampling.

    Returns:
        Perplexity (float). Lower is better.
    """
    from datasets import load_dataset

    random.seed(seed)
    model.eval()

    # Stream and collect n_samples text examples
    dataset = load_dataset(dataset_name, dataset_config,
                           split=split, streaming=True, trust_remote_code=True)
    texts = []
    for example in dataset:
        texts.append(example["text"])
        if len(texts) >= n_samples:
            break

    total_loss = 0.0
    total_tokens = 0
    loss_fn = CrossEntropyLoss(reduction="sum")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, vocab)

        # Shift: predict token i+1 from token i
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask   = attention_mask[:, 1:].contiguous()

        # Mask padding tokens
        shift_labels[shift_mask == 0] = -100

        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        n_tokens = (shift_labels != -100).sum().item()
        total_loss   += loss.item()
        total_tokens += n_tokens

    avg_nll = total_loss / max(total_tokens, 1)
    return torch.exp(torch.tensor(avg_nll)).item()


# ---------------------------------------------------------------------------
# Objective 2: Task Likelihood Margin (TLM)
# ---------------------------------------------------------------------------

def score_completion(model, tokenizer, context: str, completion: str,
                     device: str = "cuda") -> float:
    """
    Compute average per-token log probability of `completion` given `context`.

    Uses teacher-forcing over the full sequence; loss is computed only
    on the completion tokens (context tokens are masked out).

    Args:
        model:       HuggingFace CausalLM.
        tokenizer:   Matching tokenizer.
        context:     Conditioning text (prompt / question).
        completion:  Candidate continuation to score.
        device:      'cuda' or 'cpu'.

    Returns:
        Average per-token log-probability (float). Higher = more likely.
    """
    model.eval()

    full_text = context + completion
    full_enc  = tokenizer(full_text,  return_tensors="pt", truncation=True, max_length=512)
    ctx_enc   = tokenizer(context,    return_tensors="pt", truncation=True, max_length=512)

    input_ids   = full_enc["input_ids"].to(device)
    context_len = ctx_enc["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits  = outputs.logits  # (1, T, vocab)

    # Log-softmax over vocab
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (1, T, vocab)

    # Completion tokens: positions context_len .. T-1
    # We predict token t using logits at t-1
    completion_ids = input_ids[0, context_len:]          # (C,)
    completion_lp  = log_probs[0, context_len - 1: -1, :]  # (C, vocab)

    if completion_ids.numel() == 0:
        return 0.0

    token_log_probs = completion_lp.gather(
        dim=1, index=completion_ids.unsqueeze(1)
    ).squeeze(1)  # (C,)

    return token_log_probs.mean().item()


def compute_task_likelihood_margin(model, tokenizer,
                                   dataset_name: str = "hellaswag",
                                   split: str = "validation",
                                   n_samples: int = 128,
                                   device: str = "cuda",
                                   seed: int = 42) -> float:
    """
    Task Likelihood Margin (TLM) — Wee et al. 2025.

    For each multiple-choice example:
        margin = log P(correct | context)
                 - max(log P(wrong_i | context))

    Average margin over n_samples. More negative = model is more
    discriminative = better calibrated for the task.

    Args:
        model:        HuggingFace CausalLM.
        tokenizer:    Matching tokenizer.
        dataset_name: Multiple-choice HF dataset (hellaswag by default).
        split:        Dataset split.
        n_samples:    Number of examples.
        device:       'cuda' or 'cpu'.
        seed:         Random seed.

    Returns:
        Average TLM margin (float). More negative is better.
    """
    from datasets import load_dataset

    random.seed(seed)
    model.eval()

    dataset = load_dataset(dataset_name, split=split)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    margins = []
    for idx in indices:
        example = dataset[idx]

        # HellaSwag schema
        # ctx_a + ctx_b = context, endings = list of 4 candidates, label = correct index
        context    = example.get("ctx", example.get("ctx_a", "") + " " + example.get("ctx_b", ""))
        endings    = example["endings"]
        label      = int(example["label"])

        correct_score = score_completion(model, tokenizer, context, endings[label], device)
        wrong_scores  = [
            score_completion(model, tokenizer, context, endings[i], device)
            for i in range(len(endings)) if i != label
        ]
        if not wrong_scores:
            continue
        margin = correct_score - max(wrong_scores)
        margins.append(margin)

    return sum(margins) / max(len(margins), 1)
