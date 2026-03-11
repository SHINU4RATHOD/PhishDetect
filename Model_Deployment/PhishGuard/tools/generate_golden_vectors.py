#!/usr/bin/env python3
"""
Golden Vector Generator for PhishGuard Android Parity Testing.

Generates reference token IDs, attention masks, and phishing probabilities
for a set of sample URLs using the exact same model and tokenizer used in training.

IMPORTANT: This script applies the SAME normalization as Android's UrlNormalizer:
- Unicode NFKC
- Strip control characters
- Whitespace collapse
- Length cap (2048 chars)

Usage:
    python generate_golden_vectors.py \
        --model_dir ../saved_models/MiniLM_data10/best_model_epoch_018 \
        --output golden_vectors.json

Output:
    JSON array of objects, each with:
    - url: original URL
    - normalized_url: URL after normalization
    - input_ids: list of int64 token IDs (length MAX_LEN=192)
    - attention_mask: list of int64 (length MAX_LEN=192)
    - p_phishing: float probability of phishing
"""

import json
import sys
import argparse
import unicodedata
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# ── Constants (must match PhishGuardConfig.kt) ──────────────────────────────
MAX_LENGTH = 192
MAX_URL_LENGTH = 2048

# ── Sample URLs (must match PhishGuardConfig.SAMPLE_URLS) ───────────────────
SAMPLE_URLS = [
    "https://www.google.com",
    "https://github.com/login",
    "http://secure-paypal-login.suspicious-domain.xyz/verify",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "http://192.168.1.1/admin/login.php?redirect=https://bank.com",
    # Additional test vectors for edge cases
    "https://münchen.de/path",                          # IDN/punycode
    "http://gооgle.com",                                # Cyrillic 'о' confusable
    "",                                                  # Empty
    "https://example.com/" + "a" * 3000,                # Very long URL
    "https://example.com/path?q=hello+world&lang=en",   # Normal with query params
]


def normalize_url(url: str) -> str:
    """
    Apply the same normalization as Android's UrlNormalizer.normalize().
    MUST stay in sync with UrlNormalizer.kt.
    """
    if not url or not url.strip():
        return ""

    # Step 1: Unicode NFKC
    normalized = unicodedata.normalize("NFKC", url.strip())

    # Step 2: Strip control characters
    normalized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]', '', normalized)

    # Step 3: Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Step 4: Cap length
    if len(normalized) > MAX_URL_LENGTH:
        normalized = normalized[:MAX_URL_LENGTH]

    return normalized


def generate_golden_vectors(model_dir: str, output_path: str, urls: list = None):
    """Generate golden vectors for parity testing."""
    model_dir = Path(model_dir)
    
    if urls is None:
        urls = SAMPLE_URLS

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model max length: {tokenizer.model_max_length}")

    # ── Load quantized ONNX model ───────────────────────────────────────────
    model_path = model_dir / "model_quant_8bit.onnx"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(str(model_path))

    # Print I/O info for verification
    print("Model I/O:")
    for inp in session.get_inputs():
        print(f"  Input: {inp.name} shape={inp.shape} type={inp.type}")
    for out in session.get_outputs():
        print(f"  Output: {out.name} shape={out.shape} type={out.type}")

    # ── Generate vectors ────────────────────────────────────────────────────
    golden_vectors = []

    for i, url in enumerate(urls):
        print(f"\nProcessing URL {i+1}/{len(urls)}: {url[:60]}...")

        # Normalize (same as Android)
        normalized_url = normalize_url(url)

        if not normalized_url:
            print(f"  Skipping empty URL after normalization")
            continue

        # Tokenize (same as Android should produce)
        encoding = tokenizer(
            normalized_url,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        print(f"  Token count: {int(attention_mask.sum())}")
        print(f"  First 20 IDs: {input_ids[0, :20].tolist()}")

        # Run inference
        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

        logits = outputs[0][0]  # Shape [2]

        # Stable softmax
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        p_phishing = float(probs[1])
        print(f"  p_phishing: {p_phishing:.6f}")

        golden_vectors.append({
            "url": url,
            "normalized_url": normalized_url,
            "input_ids": input_ids[0].tolist(),
            "attention_mask": attention_mask[0].tolist(),
            "p_phishing": p_phishing,
        })

    # ── Save ────────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_vectors, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(golden_vectors)} golden vectors to {output_path}")
    print(f"   Copy to: app/src/main/assets/phishing/golden_vectors.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate golden vectors for PhishGuard parity testing")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../saved_models/MiniLM_data10/best_model_epoch_018",
        help="Path to best model directory containing ONNX + tokenizer"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="golden_vectors.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    generate_golden_vectors(args.model_dir, args.output)
