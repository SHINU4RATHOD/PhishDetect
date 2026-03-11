package com.phishguard.app.test

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.phishguard.app.core.PhishGuardConfig
import com.phishguard.app.core.SecureLogger
import com.phishguard.app.core.UrlNormalizer
import com.phishguard.app.data.BertWordPieceTokenizer
import com.phishguard.app.data.OnnxModelLoader
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.abs
import kotlin.math.exp

/**
 * Golden vector parity test.
 * Validates Android tokenizer + inference against Python-generated reference vectors.
 *
 * Two-stage validation:
 * - Stage 1: Exact token-ID match (zero tolerance)
 * - Stage 2: Probability tolerance (ε = 0.001)
 */
class GoldenVectorTest(
    private val context: Context,
    private val tokenizer: BertWordPieceTokenizer,
    private val modelLoader: OnnxModelLoader
) {
    private val tag = "GoldenTest"
    private val PROB_TOLERANCE = 0.001

    data class GoldenVector(
        val url: String,
        val normalized_url: String,
        val input_ids: List<Long>,
        val attention_mask: List<Long>,
        val p_phishing: Double
    )

    data class TestResult(
        val url: String,
        val stage1Pass: Boolean,          // Token ID exact match
        val stage2Pass: Boolean,          // Probability within tolerance
        val stage1Error: String?,         // Mismatch details
        val stage2Error: String?,         // Probability delta
        val androidTokenIds: List<Long>,
        val expectedTokenIds: List<Long>,
        val androidPPhishing: Double,
        val expectedPPhishing: Double
    )

    data class TestSuiteResult(
        val totalTests: Int,
        val stage1Passed: Int,
        val stage2Passed: Int,
        val allPassed: Boolean,
        val results: List<TestResult>
    )

    /**
     * Run all golden vector tests.
     *
     * @param goldenVectorsAssetPath Path to golden_vectors.json in assets
     * @return TestSuiteResult with detailed per-URL results
     */
    fun runTests(goldenVectorsAssetPath: String = "${PhishGuardConfig.ASSETS_DIR}/golden_vectors.json"): TestSuiteResult {
        SecureLogger.i(tag, "Loading golden vectors from $goldenVectorsAssetPath...")

        val vectors = loadGoldenVectors(goldenVectorsAssetPath)
        if (vectors.isEmpty()) {
            SecureLogger.e(tag, "No golden vectors found!")
            return TestSuiteResult(0, 0, 0, false, emptyList())
        }

        SecureLogger.i(tag, "Running ${vectors.size} golden vector tests...")

        val results = mutableListOf<TestResult>()
        var stage1Passed = 0
        var stage2Passed = 0

        for ((index, vector) in vectors.withIndex()) {
            SecureLogger.i(tag, "Test ${index + 1}/${vectors.size}")

            val result = testSingleVector(vector, index)
            results.add(result)

            if (result.stage1Pass) stage1Passed++
            if (result.stage2Pass) stage2Passed++
        }

        val allPassed = stage1Passed == vectors.size && stage2Passed == vectors.size

        SecureLogger.i(tag, buildString {
            append("Test suite complete: ")
            append("Stage 1 (tokenizer): $stage1Passed/${vectors.size}, ")
            append("Stage 2 (inference): $stage2Passed/${vectors.size}, ")
            append("Overall: ${if (allPassed) "PASS ✅" else "FAIL ❌"}")
        })

        return TestSuiteResult(
            totalTests = vectors.size,
            stage1Passed = stage1Passed,
            stage2Passed = stage2Passed,
            allPassed = allPassed,
            results = results
        )
    }

    private fun testSingleVector(vector: GoldenVector, testIndex: Int): TestResult {
        // Apply same normalization as Python
        val normResult = UrlNormalizer.normalize(vector.url)
        val normalizedUrl = normResult.normalizedUrl

        // Tokenize
        val tokenResult = tokenizer.tokenize(normalizedUrl)
        val androidIds = tokenResult.inputIds.toList()
        val expectedIds = vector.input_ids

        // ── Stage 1: Exact token-ID match ────────────────────────────────
        var stage1Pass = true
        var stage1Error: String? = null

        if (androidIds.size != expectedIds.size) {
            stage1Pass = false
            stage1Error = "Length mismatch: android=${androidIds.size}, expected=${expectedIds.size}"
        } else {
            for (i in androidIds.indices) {
                if (androidIds[i] != expectedIds[i]) {
                    stage1Pass = false
                    val androidTokenStr = tokenizer.idToToken(androidIds[i].toInt())
                    val expectedTokenStr = tokenizer.idToToken(expectedIds[i].toInt())
                    stage1Error = buildString {
                        append("MISMATCH at index $i: ")
                        append("expected=${expectedIds[i]}('$expectedTokenStr'), ")
                        append("got=${androidIds[i]}('$androidTokenStr')\n")
                        append("First 30 expected: ${expectedIds.take(30)}\n")
                        append("First 30 android:  ${androidIds.take(30)}")
                    }
                    break  // Report first mismatch only
                }
            }
        }

        if (!stage1Pass) {
            SecureLogger.w(tag, "Stage 1 FAIL for test $testIndex: $stage1Error", url = vector.url)
        }

        // ── Stage 2: Probability tolerance ───────────────────────────────
        var stage2Pass = false
        var stage2Error: String? = null
        var androidPPhishing = 0.0

        try {
            val logits = modelLoader.runInference(
                tokenResult.inputIds,
                tokenResult.attentionMask
            )
            val probs = stableSoftmax(logits)
            androidPPhishing = probs[PhishGuardConfig.LABEL_PHISHING].toDouble()

            val delta = abs(androidPPhishing - vector.p_phishing)
            stage2Pass = delta <= PROB_TOLERANCE

            if (!stage2Pass) {
                stage2Error = "Probability delta=${"%.6f".format(delta)} > tolerance=$PROB_TOLERANCE " +
                        "(android=${"%.6f".format(androidPPhishing)}, expected=${"%.6f".format(vector.p_phishing)})"
                SecureLogger.w(tag, "Stage 2 FAIL for test $testIndex: $stage2Error")
            }
        } catch (e: Exception) {
            stage2Error = "Inference failed: ${e.message}"
            SecureLogger.e(tag, "Stage 2 ERROR for test $testIndex", e)
        }

        return TestResult(
            url = vector.url,
            stage1Pass = stage1Pass,
            stage2Pass = stage2Pass,
            stage1Error = stage1Error,
            stage2Error = stage2Error,
            androidTokenIds = androidIds,
            expectedTokenIds = expectedIds,
            androidPPhishing = androidPPhishing,
            expectedPPhishing = vector.p_phishing
        )
    }

    private fun loadGoldenVectors(assetPath: String): List<GoldenVector> {
        return try {
            context.assets.open(assetPath).use { stream ->
                BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
                    val json = reader.readText()
                    val type = object : TypeToken<List<GoldenVector>>() {}.type
                    Gson().fromJson(json, type)
                }
            }
        } catch (e: Exception) {
            SecureLogger.e(tag, "Failed to load golden vectors", e)
            emptyList()
        }
    }

    private fun stableSoftmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - maxLogit).toDouble()).toFloat() }
        val sumExp = exps.sum()
        return FloatArray(exps.size) { exps[it] / sumExp }
    }
}
