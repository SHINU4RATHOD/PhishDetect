package com.phishguard.app.benchmark

import android.os.Build
import com.phishguard.app.core.PhishGuardConfig
import com.phishguard.app.core.SecureLogger
import com.phishguard.app.data.PhishingUrlDetector
import com.phishguard.app.domain.BenchmarkResult
import com.phishguard.app.domain.DetectionResult

/**
 * Benchmark runner for phishing URL detection.
 *
 * Runs:
 * - N warmup scans (full pipeline: normalize → tokenize → inference → softmax)
 * - M timed scans with per-stage timing
 * - Reports p50, p90, mean, min, max latency
 * - Logs device info + execution provider
 */
class BenchmarkRunner(private val detector: PhishingUrlDetector) {

    private val tag = "Benchmark"

    /**
     * Run benchmark with default URLs and settings.
     *
     * @param urls List of URLs to benchmark against
     * @param warmupRuns Number of warmup runs (default from config)
     * @param timedRuns Number of timed runs (default from config)
     * @param onProgress Callback for progress updates (0.0 to 1.0)
     * @return BenchmarkResult with all statistics
     */
    fun run(
        urls: List<String> = PhishGuardConfig.SAMPLE_URLS,
        warmupRuns: Int = PhishGuardConfig.BENCHMARK_WARMUP_RUNS,
        timedRuns: Int = PhishGuardConfig.BENCHMARK_TIMED_RUNS,
        onProgress: ((Float) -> Unit)? = null
    ): BenchmarkResult {
        val deviceInfo = getDeviceInfo()
        SecureLogger.i(tag, "Starting benchmark: warmup=$warmupRuns, timed=$timedRuns")
        SecureLogger.i(tag, "Device: $deviceInfo")

        val totalOps = warmupRuns + timedRuns
        var currentOp = 0

        // Phase 1: Warmup (full pipeline to populate caches)
        SecureLogger.i(tag, "Phase 1: Warmup ($warmupRuns runs)...")
        for (i in 0 until warmupRuns) {
            val url = urls[i % urls.size]
            detector.predict(url)
            currentOp++
            onProgress?.invoke(currentOp.toFloat() / totalOps)
        }

        // Phase 2: Timed runs
        SecureLogger.i(tag, "Phase 2: Timed ($timedRuns runs)...")
        val results = mutableListOf<DetectionResult>()

        for (i in 0 until timedRuns) {
            val url = urls[i % urls.size]
            val result = detector.predict(url)
            results.add(result)
            currentOp++
            onProgress?.invoke(currentOp.toFloat() / totalOps)
        }

        // Compute statistics from successful results
        val successfulResults = results.filter { it.isSuccess }

        if (successfulResults.isEmpty()) {
            SecureLogger.e(tag, "All benchmark runs failed!")
            return BenchmarkResult(
                totalRuns = timedRuns,
                warmupRuns = warmupRuns,
                meanTokenizeMs = 0.0,
                meanInferenceMs = 0.0,
                meanTotalMs = 0.0,
                p50TotalMs = 0.0,
                p90TotalMs = 0.0,
                minTotalMs = 0.0,
                maxTotalMs = 0.0,
                executionProvider = detector.executionProvider,
                deviceInfo = deviceInfo,
                results = results
            )
        }

        val tokenizeTimes = successfulResults.map { it.tokenizeTimeMs }
        val inferenceTimes = successfulResults.map { it.inferenceTimeMs }
        val totalTimes = successfulResults.map { it.totalTimeMs }

        val sortedTotal = totalTimes.sorted()

        val result = BenchmarkResult(
            totalRuns = successfulResults.size,
            warmupRuns = warmupRuns,
            meanTokenizeMs = tokenizeTimes.average(),
            meanInferenceMs = inferenceTimes.average(),
            meanTotalMs = totalTimes.average(),
            p50TotalMs = percentile(sortedTotal, 50),
            p90TotalMs = percentile(sortedTotal, 90),
            minTotalMs = sortedTotal.first(),
            maxTotalMs = sortedTotal.last(),
            executionProvider = detector.executionProvider,
            deviceInfo = deviceInfo,
            results = results
        )

        SecureLogger.i(tag, buildString {
            append("Benchmark complete: ")
            append("mean=${"%.1f".format(result.meanTotalMs)}ms, ")
            append("p50=${"%.1f".format(result.p50TotalMs)}ms, ")
            append("p90=${"%.1f".format(result.p90TotalMs)}ms, ")
            append("EP=${result.executionProvider}")
        })

        return result
    }

    /**
     * Calculate percentile from a sorted list.
     */
    private fun percentile(sortedValues: List<Double>, p: Int): Double {
        if (sortedValues.isEmpty()) return 0.0
        val index = (p / 100.0 * (sortedValues.size - 1)).let {
            val lower = it.toInt()
            val fraction = it - lower
            if (lower + 1 < sortedValues.size) {
                sortedValues[lower] + fraction * (sortedValues[lower + 1] - sortedValues[lower])
            } else {
                sortedValues[lower]
            }
        }
        return index
    }

    /**
     * Get formatted device info string.
     */
    private fun getDeviceInfo(): String {
        return buildString {
            append("${Build.MANUFACTURER} ${Build.MODEL}")
            append(" | Android ${Build.VERSION.RELEASE}")
            append(" (API ${Build.VERSION.SDK_INT})")
            append(" | ABI: ${Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"}")
        }
    }
}
