package com.phishguard.app.data

import android.content.Context
import com.phishguard.app.core.PhishGuardConfig
import com.phishguard.app.core.SecureLogger
import com.phishguard.app.core.UrlNormalizer
import com.phishguard.app.domain.DetectionResult
import kotlin.math.exp
import kotlin.math.max

/**
 * Main phishing URL detector.
 * Orchestrates: normalization → tokenization → ONNX inference → softmax → threshold.
 *
 * Usage:
 * ```
 * val detector = PhishingUrlDetector(context)
 * detector.initialize()
 * val result = detector.predict("https://suspicious-site.xyz/login")
 * ```
 */
class PhishingUrlDetector(private val context: Context) {

    private val tag = "Detector"

    private var _tokenizer: BertWordPieceTokenizer? = null
    private var _modelLoader: OnnxModelLoader? = null

    // Exposed for golden vector parity tests
    val tokenizer: BertWordPieceTokenizer? get() = _tokenizer
    val modelLoader: OnnxModelLoader? get() = _modelLoader

    val isReady: Boolean
        get() = tokenizer != null && modelLoader?.isLoaded == true

    val executionProvider: String
        get() = modelLoader?.executionProvider ?: "UNKNOWN"

    /**
     * Initialize tokenizer and model.
     * Should be called once during app startup (e.g., in ViewModel init).
     *
     * @param enableNnapi Whether to attempt NNAPI acceleration
     * @return True if both tokenizer and model loaded successfully
     */
    fun initialize(enableNnapi: Boolean = true): Boolean {
        return try {
            SecureLogger.i(tag, "Initializing PhishingUrlDetector...")

            // Load tokenizer (from vocab.txt)
            _tokenizer = BertWordPieceTokenizer(context)

            // Load ONNX model
            _modelLoader = OnnxModelLoader(context)
            val modelOk = _modelLoader!!.initialize(enableNnapi)

            if (!modelOk) {
                SecureLogger.e(tag, "Model initialization failed")
                return false
            }

            SecureLogger.i(tag, "Detector ready. EP=${_modelLoader!!.executionProvider}")
            true
        } catch (e: Exception) {
            SecureLogger.e(tag, "Initialization failed", e)
            false
        }
    }

    /**
     * Run end-to-end prediction on a raw URL.
     *
     * @param url Raw URL string (will be normalized internally)
     * @return DetectionResult with verdict, probability, and debug info
     */
    fun predict(url: String): DetectionResult {
        if (!isReady) {
            return DetectionResult.error("Detector not initialized")
        }

        val startTime = System.nanoTime()

        try {
            // Step 1: Validate input
            if (url.isBlank()) {
                return DetectionResult.error("Empty URL")
            }

            // Step 2: Normalize URL (same pipeline as Python golden vectors)
            val normResult = UrlNormalizer.normalize(url)
            val normalizedUrl = normResult.normalizedUrl

            if (normalizedUrl.isEmpty()) {
                return DetectionResult.error("URL empty after normalization")
            }

            // Step 3: Tokenize
            val tokenizeStartTime = System.nanoTime()
            val tokenResult = _tokenizer!!.tokenize(normalizedUrl)
            val tokenizeTimeMs = (System.nanoTime() - tokenizeStartTime) / 1_000_000.0

            // Step 4: Run inference
            val inferenceStartTime = System.nanoTime()
            val logits = _modelLoader!!.runInference(
                tokenResult.inputIds,
                tokenResult.attentionMask
            )
            val inferenceTimeMs = (System.nanoTime() - inferenceStartTime) / 1_000_000.0

            // Step 5: Stable softmax → probabilities
            val probabilities = stableSoftmax(logits)
            val pPhishing = probabilities[PhishGuardConfig.LABEL_PHISHING]
            val pBenign = probabilities[PhishGuardConfig.LABEL_BENIGN]

            // Step 6: Apply threshold
            val isPhishing = pPhishing >= PhishGuardConfig.PHISHING_THRESHOLD
            val label = if (isPhishing) {
                PhishGuardConfig.LABEL_PHISHING_TEXT
            } else {
                PhishGuardConfig.LABEL_BENIGN_TEXT
            }

            val totalTimeMs = (System.nanoTime() - startTime) / 1_000_000.0

            SecureLogger.i(
                tag,
                "Prediction: $label (p=${"%.4f".format(pPhishing)}, " +
                "latency=${"%.1f".format(totalTimeMs)}ms)",
                url = url
            )

            return DetectionResult(
                url = url,
                normalizedUrl = normalizedUrl,
                label = label,
                isPhishing = isPhishing,
                phishingProbability = pPhishing,
                benignProbability = pBenign,
                threshold = PhishGuardConfig.PHISHING_THRESHOLD,
                tokenCount = tokenResult.tokenCount,
                maxLength = PhishGuardConfig.MAX_LENGTH,
                tokenizeTimeMs = tokenizeTimeMs,
                inferenceTimeMs = inferenceTimeMs,
                totalTimeMs = totalTimeMs,
                executionProvider = _modelLoader!!.executionProvider,
                modelFileName = PhishGuardConfig.MODEL_FILENAME,
                normalizationWarnings = normResult.warnings,
                containsPunycode = normResult.containsPunycode,
                containsSuspiciousUnicode = normResult.containsSuspiciousUnicode,
                error = null
            )
        } catch (e: Exception) {
            val totalTimeMs = (System.nanoTime() - startTime) / 1_000_000.0
            SecureLogger.e(tag, "Prediction failed", e, url = url)
            return DetectionResult.error("Prediction failed: ${e.message}", totalTimeMs)
        }
    }

    /**
     * Numerically stable softmax.
     * Subtracts max logit before exp() to prevent overflow.
     */
    private fun stableSoftmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - maxLogit).toDouble()).toFloat() }
        val sumExp = exps.sum()
        return FloatArray(exps.size) { exps[it] / sumExp }
    }

    /**
     * Release resources.
     */
    fun close() {
        _modelLoader?.close()
        _modelLoader = null
        _tokenizer = null
        SecureLogger.i(tag, "Detector closed")
    }
}
