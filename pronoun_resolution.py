# =============================================================
# Pronoun Resolution — Zero-Shot with Pretrained BERT MLM
# =============================================================
# Strategy (Pseudo-Log-Likelihood):
#   For each candidate, replace the pronoun with that candidate,
#   then mask every token of the candidate one at a time and sum
#   the log-probs BERT assigns to those tokens.
#   The candidate whose substituted sentence scores higher wins.
#
# No fine-tuning — uses bert-base-uncased straight from HuggingFace.
# =============================================================

# pip install transformers datasets torch scikit-learn tqdm

import math
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load pretrained BERT (MLM head) ──────────────────────
print("Loading pretrained bert-base-uncased …")
MODEL_NAME = "bert-base-uncased"
tokenizer  = BertTokenizer.from_pretrained(MODEL_NAME)
model      = BertForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()
print("Model ready — no training needed.\n")

# ── 2. Scoring function ──────────────────────────────────────
def sentence_log_prob(sentence: str, candidate: str) -> float:
    """
    Returns the sum of log P(token_i | context) for every subword token
    of `candidate` inside `sentence`, using BERT's masked language model.

    Approach:
      • Tokenize `sentence` (already contains the candidate).
      • Locate the token span belonging to `candidate`.
      • Mask those tokens one at a time; accumulate log-probs.
    """
    # Encode full sentence
    enc = tokenizer(sentence, return_tensors="pt")
    all_ids = enc["input_ids"][0]          # [seq_len]
    seq_len = all_ids.size(0)

    # Find which token positions correspond to `candidate`
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)

    # Locate the span of candidate_ids inside all_ids
    span_start = None
    for i in range(seq_len - len(candidate_ids) + 1):
        if all_ids[i: i + len(candidate_ids)].tolist() == candidate_ids:
            span_start = i
            break

    # Fallback: score just the first token at the pronoun position
    if span_start is None:
        # If candidate isn't found verbatim (e.g., casing mismatch), fall
        # back to scoring its first subword token at position 1.
        candidate_ids = tokenizer.encode(
            candidate.lower(), add_special_tokens=False
        )
        span_start = 1

    span_end = span_start + len(candidate_ids)   # exclusive

    total_log_prob = 0.0
    for pos in range(span_start, span_end):
        # Build a copy with this position masked
        masked_ids = all_ids.clone()
        masked_ids[pos] = tokenizer.mask_token_id

        input_dict = {
            "input_ids":      masked_ids.unsqueeze(0),
            "attention_mask": enc["attention_mask"],
        }
        with torch.no_grad():
            logits = model(**input_dict).logits          # [1, seq_len, vocab]

        log_probs = torch.log_softmax(logits[0, pos], dim=-1)
        total_log_prob += log_probs[all_ids[pos]].item()

    return total_log_prob


# ── 3. Predict function ──────────────────────────────────────
def predict(sentence: str, pronoun: str, option1: str, option2: str,
            verbose: bool = True) -> str:
    """
    Given a sentence with a pronoun and two candidate antecedents,
    returns whichever candidate is more likely according to BERT MLM.
    """
    sent1 = sentence.replace(pronoun, option1)
    sent2 = sentence.replace(pronoun, option2)

    score1 = sentence_log_prob(sent1, option1)
    score2 = sentence_log_prob(sent2, option2)

    prediction = option1 if score1 > score2 else option2
    if verbose:
        print(f"  Sentence : {sentence}")
        print(f"  Option 1 : {option1:<15} score = {score1:.4f}")
        print(f"  Option 2 : {option2:<15} score = {score2:.4f}")
        print(f"  → Predicted : {prediction}\n")
    return prediction


# ── 4. Evaluate on the DPR test set ─────────────────────────
print("Loading DPR dataset …")
ds = load_dataset("community-datasets/definite_pronoun_resolution")

def evaluate(split: str = "test"):
    data      = ds[split]
    labels    = []
    preds     = []

    for ex in tqdm(data, desc=f"Evaluating [{split}]"):
        sentence  = ex["sentence"]
        pronoun   = ex["pronoun"]
        option1   = ex["candidates"][0]
        option2   = ex["candidates"][1]
        label     = ex["label"]          # 0 → option1 is correct, 1 → option2

        pred_text = predict(sentence, pronoun, option1, option2, verbose=False)
        pred_idx  = 0 if pred_text == option1 else 1

        labels.append(label)
        preds.append(pred_idx)

    acc = accuracy_score(labels, preds)
    print(f"\n{'='*50}")
    print(f"  Split      : {split}")
    print(f"  Samples    : {len(labels)}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(labels, preds,
                                target_names=["option1", "option2"]))
    return acc

print()
evaluate("test")


# ── 5. Demo predictions ──────────────────────────────────────
print("\n" + "="*50)
print("  Custom Predictions")
print("="*50 + "\n")

predict("Rahul met Aman and he smiled",             "he",  "Rahul",  "Aman")
predict("Priya called Neha because she was late",   "she", "Priya",  "Neha")
predict("Ramesh helped Suresh because he was kind", "he",  "Ramesh", "Suresh")
