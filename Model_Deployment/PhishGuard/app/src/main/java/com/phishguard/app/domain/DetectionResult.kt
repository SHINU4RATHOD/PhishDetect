package com.phishguard.app.domain

/**
 * Data class representing the result of a phishing URL detection scan.
 * Contains all information needed for UI display, logging, and parity testing.
 */
data class DetectionResult(
    val url: String,
    val normalizedUrl: String,
    val label: String,                    // "SAFE" or "PHISHING"
    val isPhishing: Boolean,
    val phishingProbability: Float,        // p(phishing) ∈ [0, 1]
    val benignProbability: Float,          // p(benign) ∈ [0, 1]
    val threshold: Double,                 // Decision threshold (0.59)
    val tokenCount: Int,                   // Actual tokens before padding
    val maxLength: Int,                    // MAX_LEN used
    val tokenizeTimeMs: Double,            // Tokenization latency
    val inferenceTimeMs: Double,           // ORT inference latency
    val totalTimeMs: Double,               // End-to-end latency
    val executionProvider: String,         // "CPU" or "NNAPI"
    val modelFileName: String,             // For display
    val normalizationWarnings: List<String>,
    val containsPunycode: Boolean,
    val containsSuspiciousUnicode: Boolean,
    val error: String?                     // Non-null if error occurred
) {
    val isError: Boolean get() = error != null
    val isSuccess: Boolean get() = error == null

    /** Formatted probability for display (e.g., "87.3%") */
    val phishingPercentage: String
        get() = "${"%.1f".format(phishingProbability * 100)}%"

    /** Formatted latency for display (e.g., "23.4 ms") */
    val formattedLatency: String
        get() = "${"%.1f".format(totalTimeMs)} ms"

    /** Short display URL (truncated) */
    val displayUrl: String
        get() = if (url.length > 80) url.take(77) + "..." else url

    /** Timestamp of scan */
    val timestamp: Long = System.currentTimeMillis()

    companion object {
        fun error(message: String, totalTimeMs: Double = 0.0): DetectionResult {
            return DetectionResult(
                url = "",
                normalizedUrl = "",
                label = "ERROR",
                isPhishing = false,
                phishingProbability = 0f,
                benignProbability = 0f,
                threshold = 0.0,
                tokenCount = 0,
                maxLength = 0,
                tokenizeTimeMs = 0.0,
                inferenceTimeMs = 0.0,
                totalTimeMs = totalTimeMs,
                executionProvider = "NONE",
                modelFileName = "",
                normalizationWarnings = emptyList(),
                containsPunycode = false,
                containsSuspiciousUnicode = false,
                error = message
            )
        }
    }
}

/**
 * Benchmark results for a series of scans.
 */
data class BenchmarkResult(
    val totalRuns: Int,
    val warmupRuns: Int,
    val meanTokenizeMs: Double,
    val meanInferenceMs: Double,
    val meanTotalMs: Double,
    val p50TotalMs: Double,
    val p90TotalMs: Double,
    val minTotalMs: Double,
    val maxTotalMs: Double,
    val executionProvider: String,
    val deviceInfo: String,
    val results: List<DetectionResult>
)
