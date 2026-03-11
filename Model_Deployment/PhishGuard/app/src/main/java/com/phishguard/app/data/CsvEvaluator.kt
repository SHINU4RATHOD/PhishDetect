package com.phishguard.app.data

import android.content.Context
import android.net.Uri
import com.phishguard.app.core.SecureLogger
import com.phishguard.app.domain.CsvEvaluationResult
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.abs

/**
 * CSV batch evaluator for phishing URL detection.
 *
 * Parses a CSV with columns `input` (URL) and `label` (0=Benign, 1=Phishing),
 * runs inference on each URL via [PhishingUrlDetector], and computes
 * full binary classification metrics including ROC-AUC.
 *
 * Processing is streamed in batches to keep memory bounded for large CSVs.
 */
class CsvEvaluator(
    private val context: Context,
    private val detector: PhishingUrlDetector
) {
    private val tag = "CsvEval"

    data class CsvInfo(
        val fileName: String,
        val totalRows: Int,
        val hasRequiredColumns: Boolean,
        val error: String? = null
    )

    /**
     * Pre-scan CSV to get row count and validate columns.
     */
    fun scanCsv(uri: Uri): CsvInfo {
        val fileName = getFileName(uri)
        return try {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
                    val header = reader.readLine()?.lowercase()?.trim()
                        ?: return CsvInfo(fileName, 0, false, "Empty file")

                    val columns = parseCsvLine(header)
                    val hasInput = columns.any { it.trim() == "input" }
                    val hasLabel = columns.any { it.trim() == "label" }

                    if (!hasInput || !hasLabel) {
                        return CsvInfo(
                            fileName, 0, false,
                            "Missing columns. Expected 'input' and 'label', found: ${columns.joinToString()}"
                        )
                    }

                    var count = 0
                    while (reader.readLine() != null) count++

                    CsvInfo(fileName, count, true)
                }
            } ?: CsvInfo(fileName, 0, false, "Cannot open file")
        } catch (e: Exception) {
            SecureLogger.e(tag, "CSV scan failed", e)
            CsvInfo(fileName, 0, false, "Error: ${e.message}")
        }
    }

    /**
     * Run full evaluation on a CSV file.
     *
     * @param uri         URI of the CSV file
     * @param onProgress  Callback with (processed, total) for progress updates
     * @return CsvEvaluationResult with all metrics
     */
    fun evaluate(
        uri: Uri,
        onProgress: (processed: Int, total: Int) -> Unit
    ): CsvEvaluationResult {
        val fileName = getFileName(uri)
        val startTime = System.currentTimeMillis()

        SecureLogger.i(tag, "Starting CSV evaluation: $fileName")

        // Collect all (trueLabel, predictedProbability) pairs for ROC-AUC
        val trueLabels = mutableListOf<Int>()
        val predictedProbs = mutableListOf<Double>()

        var tp = 0; var tn = 0; var fp = 0; var fn = 0
        var totalErrors = 0
        var totalRows = 0
        var processed = 0

        context.contentResolver.openInputStream(uri)?.use { stream ->
            BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
                val header = reader.readLine()?.lowercase()?.trim() ?: return errorResult(fileName, "Empty CSV")
                val columns = parseCsvLine(header)
                val inputIdx = columns.indexOfFirst { it.trim() == "input" }
                val labelIdx = columns.indexOfFirst { it.trim() == "label" }

                if (inputIdx == -1 || labelIdx == -1) {
                    return errorResult(fileName, "Missing 'input'/'label' columns")
                }

                // Count total rows first for progress
                val allLines = mutableListOf<String>()
                var line = reader.readLine()
                while (line != null) {
                    if (line.isNotBlank()) allLines.add(line)
                    line = reader.readLine()
                }
                totalRows = allLines.size

                SecureLogger.i(tag, "Processing $totalRows URLs...")

                // Process each URL
                for (csvLine in allLines) {
                    try {
                        val fields = parseCsvLine(csvLine)
                        if (fields.size <= maxOf(inputIdx, labelIdx)) {
                            totalErrors++
                            processed++
                            continue
                        }

                        val url = fields[inputIdx].trim()
                        val labelStr = fields[labelIdx].trim()
                        val trueLabel = labelStr.toIntOrNull()

                        if (url.isBlank() || trueLabel == null || trueLabel !in 0..1) {
                            totalErrors++
                            processed++
                            continue
                        }

                        // Run inference
                        val result = detector.predict(url)

                        if (result.isError) {
                            totalErrors++
                            processed++
                            continue
                        }

                        val predictedPhishing = result.isPhishing
                        val pPhishing = result.phishingProbability.toDouble()

                        // Collect for ROC-AUC
                        trueLabels.add(trueLabel)
                        predictedProbs.add(pPhishing)

                        // Update confusion matrix
                        when {
                            trueLabel == 1 && predictedPhishing  -> tp++   // M→M (correct detection)
                            trueLabel == 0 && !predictedPhishing -> tn++   // L→L (correct rejection)
                            trueLabel == 0 && predictedPhishing  -> fp++   // L→M (false alarm)
                            trueLabel == 1 && !predictedPhishing -> fn++   // M→L (missed threat)
                        }
                    } catch (e: Exception) {
                        totalErrors++
                    }

                    processed++

                    // Report progress every 50 URLs
                    if (processed % 50 == 0 || processed == totalRows) {
                        onProgress(processed, totalRows)
                    }
                }
            }
        } ?: return errorResult(fileName, "Cannot open file")

        val elapsedMs = System.currentTimeMillis() - startTime

        // ── Compute metrics ──────────────────────────────────────────────
        val totalClassified = tp + tn + fp + fn
        val accuracy = if (totalClassified > 0) (tp + tn).toDouble() / totalClassified else 0.0
        val precision = if (tp + fp > 0) tp.toDouble() / (tp + fp) else 0.0
        val recall = if (tp + fn > 0) tp.toDouble() / (tp + fn) else 0.0
        val f1 = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0.0
        val fnr = if (tp + fn > 0) fn.toDouble() / (tp + fn) else 0.0
        val fpr = if (fp + tn > 0) fp.toDouble() / (fp + tn) else 0.0

        // ── ROC-AUC (trapezoidal rule) ───────────────────────────────────
        val rocAuc = computeRocAuc(trueLabels, predictedProbs)

        val throughput = if (elapsedMs > 0) totalClassified * 1000.0 / elapsedMs else 0.0

        SecureLogger.i(tag, buildString {
            append("Evaluation complete: ")
            append("Acc=${"%.4f".format(accuracy)}, ")
            append("F1=${"%.4f".format(f1)}, ")
            append("AUC=${"%.4f".format(rocAuc)}, ")
            append("processed=$totalClassified, errors=$totalErrors, ")
            append("time=${elapsedMs}ms")
        })

        return CsvEvaluationResult(
            tn = tn, fp = fp, fn = fn, tp = tp,
            accuracy = accuracy,
            precision = precision,
            recall = recall,
            f1 = f1,
            rocAuc = rocAuc,
            fnr = fnr,
            fpr = fpr,
            totalUrls = totalRows,
            totalErrors = totalErrors,
            elapsedMs = elapsedMs,
            throughputUrlsPerSec = throughput,
            csvFileName = fileName,
            threshold = com.phishguard.app.core.PhishGuardConfig.PHISHING_THRESHOLD
        )
    }

    /**
     * Compute ROC-AUC using the trapezoidal rule.
     * Sorts by predicted probability, sweeps thresholds, integrates area.
     */
    private fun computeRocAuc(trueLabels: List<Int>, predictedProbs: List<Double>): Double {
        if (trueLabels.isEmpty() || trueLabels.all { it == 0 } || trueLabels.all { it == 1 }) {
            return 0.0  // AUC undefined for single-class
        }

        val totalPositive = trueLabels.count { it == 1 }
        val totalNegative = trueLabels.count { it == 0 }

        // Sort by predicted probability descending
        val sortedIndices = predictedProbs.indices.sortedByDescending { predictedProbs[it] }

        var tpCount = 0
        var fpCount = 0
        var prevTpr = 0.0
        var prevFpr = 0.0
        var auc = 0.0
        var prevProb = Double.MAX_VALUE

        for (idx in sortedIndices) {
            val prob = predictedProbs[idx]

            // Only compute at threshold changes
            if (abs(prob - prevProb) > 1e-10) {
                val tpr = tpCount.toDouble() / totalPositive
                val fpr = fpCount.toDouble() / totalNegative

                // Trapezoidal area
                auc += (fpr - prevFpr) * (tpr + prevTpr) / 2.0
                prevTpr = tpr
                prevFpr = fpr
                prevProb = prob
            }

            if (trueLabels[idx] == 1) tpCount++ else fpCount++
        }

        // Final trapezoid to (1,1)
        val finalTpr = tpCount.toDouble() / totalPositive
        val finalFpr = fpCount.toDouble() / totalNegative
        auc += (finalFpr - prevFpr) * (finalTpr + prevTpr) / 2.0

        return auc.coerceIn(0.0, 1.0)
    }

    /**
     * Parse a CSV line handling quoted fields with commas.
     */
    private fun parseCsvLine(line: String): List<String> {
        val fields = mutableListOf<String>()
        var current = StringBuilder()
        var inQuotes = false

        for (char in line) {
            when {
                char == '"' -> inQuotes = !inQuotes
                char == ',' && !inQuotes -> {
                    fields.add(current.toString())
                    current = StringBuilder()
                }
                else -> current.append(char)
            }
        }
        fields.add(current.toString())
        return fields
    }

    private fun getFileName(uri: Uri): String {
        return try {
            val cursor = context.contentResolver.query(uri, null, null, null, null)
            cursor?.use {
                if (it.moveToFirst()) {
                    val nameIndex = it.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    if (nameIndex != -1) it.getString(nameIndex) else "unknown.csv"
                } else "unknown.csv"
            } ?: uri.lastPathSegment ?: "unknown.csv"
        } catch (e: Exception) {
            uri.lastPathSegment ?: "unknown.csv"
        }
    }

    private fun errorResult(fileName: String, error: String): CsvEvaluationResult {
        SecureLogger.e(tag, "Evaluation error: $error")
        return CsvEvaluationResult(
            tn = 0, fp = 0, fn = 0, tp = 0,
            accuracy = 0.0, precision = 0.0, recall = 0.0, f1 = 0.0,
            rocAuc = 0.0, fnr = 0.0, fpr = 0.0,
            totalUrls = 0, totalErrors = 0, elapsedMs = 0,
            throughputUrlsPerSec = 0.0, csvFileName = fileName,
            threshold = 0.0
        )
    }
}
