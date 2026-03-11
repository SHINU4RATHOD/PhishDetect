package com.phishguard.app.domain

/**
 * Complete evaluation result for a CSV batch evaluation.
 * Stores all binary classification metrics + confusion matrix + performance stats.
 *
 * Label mapping:
 *   L = Legitimate (Benign, label=0)
 *   M = Malicious  (Phishing, label=1)
 *
 * Confusion matrix:
 *   tn (L→L) = Predicted Benign, Actually Benign     (correct rejection)
 *   fp (L→M) = Predicted Phishing, Actually Benign    (false alarm)
 *   fn (M→L) = Predicted Benign, Actually Phishing    (missed threat!)
 *   tp (M→M) = Predicted Phishing, Actually Phishing  (correct detection)
 */
data class CsvEvaluationResult(
    // ── Confusion Matrix ─────────────────────────────────────────────────
    val tn: Int,      // True Negative  (L→L)
    val fp: Int,      // False Positive  (L→M)
    val fn: Int,      // False Negative  (M→L)
    val tp: Int,      // True Positive   (M→M)

    // ── Classification Metrics ───────────────────────────────────────────
    val accuracy: Double,
    val precision: Double,
    val recall: Double,
    val f1: Double,
    val rocAuc: Double,
    val fnr: Double,  // False Negative Rate = fn / (fn + tp)
    val fpr: Double,  // False Positive Rate = fp / (fp + tn)

    // ── Performance Stats ────────────────────────────────────────────────
    val totalUrls: Int,
    val totalErrors: Int,
    val elapsedMs: Long,
    val throughputUrlsPerSec: Double,

    // ── CSV Info ─────────────────────────────────────────────────────────
    val csvFileName: String,
    val threshold: Double
) {
    val totalProcessed: Int get() = tn + fp + fn + tp
    val totalPositives: Int get() = tp + fn   // Actually phishing
    val totalNegatives: Int get() = tn + fp   // Actually benign

    /** Formatted accuracy for display (e.g., "97.32%") */
    val accuracyDisplay: String get() = "${"%.2f".format(accuracy * 100)}%"
    val precisionDisplay: String get() = "${"%.2f".format(precision * 100)}%"
    val recallDisplay: String get() = "${"%.2f".format(recall * 100)}%"
    val f1Display: String get() = "${"%.4f".format(f1)}"
    val rocAucDisplay: String get() = "${"%.4f".format(rocAuc)}"
    val fnrDisplay: String get() = "${"%.4f".format(fnr)}"
    val fprDisplay: String get() = "${"%.4f".format(fpr)}"
    val throughputDisplay: String get() = "${"%.1f".format(throughputUrlsPerSec)} URLs/s"
    val elapsedDisplay: String
        get() {
            val seconds = elapsedMs / 1000
            return if (seconds > 60) "${seconds / 60}m ${seconds % 60}s" else "${seconds}s"
        }
}
