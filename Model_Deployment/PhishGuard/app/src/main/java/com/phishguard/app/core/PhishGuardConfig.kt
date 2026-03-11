package com.phishguard.app.core

/**
 * PhishGuard configuration constants.
 * All values derived from deployment_metadata.json + final_results.json
 * (single source of truth from the MiniLM training pipeline).
 */
object PhishGuardConfig {

    // ── Model ────────────────────────────────────────────────────────────────
    const val MODEL_FILENAME = "model_quant_8bit.onnx"
    const val VOCAB_FILENAME = "vocab.txt"
    const val CONFIG_FILENAME = "phishguard_config.json"
    const val ASSETS_DIR = "phishing"

    // ── Inference Contract ───────────────────────────────────────────────────
    const val MAX_LENGTH = 192
    const val NUM_CLASSES = 2
    const val PHISHING_THRESHOLD = 0.59

    // Label mapping: index 0 = benign, index 1 = phishing
    const val LABEL_BENIGN = 0
    const val LABEL_PHISHING = 1
    const val LABEL_BENIGN_TEXT = "SAFE"
    const val LABEL_PHISHING_TEXT = "PHISHING"

    // Expected ONNX I/O names (cross-checked; discovered dynamically at runtime)
    const val EXPECTED_INPUT_IDS_NAME = "input_ids"
    const val EXPECTED_ATTENTION_MASK_NAME = "attention_mask"
    const val EXPECTED_OUTPUT_NAME = "logits"

    // ── Tokenizer Special Tokens ─────────────────────────────────────────────
    const val PAD_TOKEN_ID = 0L
    const val UNK_TOKEN_ID = 100L
    const val CLS_TOKEN_ID = 101L
    const val SEP_TOKEN_ID = 102L
    const val VOCAB_SIZE = 30522

    // ── URL Security ─────────────────────────────────────────────────────────
    const val MAX_URL_LENGTH = 2048  // Cap extremely long URLs
    const val MAX_URL_DISPLAY_LENGTH = 80

    // ── Benchmark ────────────────────────────────────────────────────────────
    const val BENCHMARK_WARMUP_RUNS = 5
    const val BENCHMARK_TIMED_RUNS = 20

    // ── ONNX Runtime Tuning ──────────────────────────────────────────────────
    const val ORT_INTRA_OP_THREADS = 2
    const val ORT_INTER_OP_THREADS = 1
    const val NNAPI_MIN_API_LEVEL = 27  // NNAPI meaningful from API 27+

    // ── Sample URLs for smoke test ───────────────────────────────────────────
    val SAMPLE_URLS = listOf(
        "https://www.google.com",
        "https://github.com/login",
        "http://secure-paypal-login.suspicious-domain.xyz/verify",
        "https://www.amazon.com/dp/B08N5WRWNW",
        "http://192.168.1.1/admin/login.php?redirect=https://bank.com"
    )
}
