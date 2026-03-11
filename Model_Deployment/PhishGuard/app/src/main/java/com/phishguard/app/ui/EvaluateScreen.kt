package com.phishguard.app.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.phishguard.app.domain.CsvEvaluationResult
import com.phishguard.app.ui.theme.*

@Composable
fun EvaluateScreen(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    val evalState by viewModel.evaluationState.collectAsState()

    val csvPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { viewModel.onCsvSelected(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp)
    ) {
        Spacer(modifier = Modifier.height(16.dp))

        // ── Header ───────────────────────────────────────────────────────
        Text(
            text = "CSV Evaluation",
            style = MaterialTheme.typography.headlineLarge,
            color = MaterialTheme.colorScheme.onBackground
        )
        Text(
            text = "Upload labeled URLs • Full metrics dashboard",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(24.dp))

        // ── Upload Section ───────────────────────────────────────────────
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    Icons.Default.Add,
                    contentDescription = null,
                    modifier = Modifier.size(40.dp),
                    tint = AccentPurple
                )
                Spacer(modifier = Modifier.height(8.dp))

                if (evalState.csvFileName.isNotEmpty()) {
                    Text(
                        text = evalState.csvFileName,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "${evalState.csvTotalRows} URLs detected",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                } else {
                    Text(
                        text = "Select a CSV file with 'input' and 'label' columns",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    // Select CSV button
                    OutlinedButton(
                        onClick = { csvPicker.launch(arrayOf("text/*", "application/csv")) },
                        modifier = Modifier.weight(1f),
                        enabled = !evalState.isRunning,
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Default.Search, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Select CSV")
                    }

                    // Run button
                    Button(
                        onClick = { viewModel.runEvaluation() },
                        modifier = Modifier.weight(1f),
                        enabled = uiState.isReady && evalState.csvUri != null && !evalState.isRunning,
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = AccentPurple)
                    ) {
                        if (evalState.isRunning) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(16.dp),
                                color = Color.White,
                                strokeWidth = 2.dp
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                        } else {
                            Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                        }
                        Text("Evaluate")
                    }
                }
            }
        }

        // ── Progress ─────────────────────────────────────────────────────
        AnimatedVisibility(visible = evalState.isRunning) {
            Column(modifier = Modifier.padding(top = 16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Processing...",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Text(
                        text = "${evalState.processed}/${evalState.csvTotalRows}",
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        color = AccentPurple
                    )
                }
                Spacer(modifier = Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { evalState.progress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp)),
                    color = AccentPurple,
                    trackColor = MaterialTheme.colorScheme.surfaceVariant
                )
            }
        }

        // ── Error ────────────────────────────────────────────────────────
        evalState.error?.let { error ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 16.dp),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = DangerRedSurface)
            ) {
                Row(
                    modifier = Modifier.padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.Error, contentDescription = null, tint = DangerRed, modifier = Modifier.size(20.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(text = error, style = MaterialTheme.typography.bodySmall, color = DangerRed)
                }
            }
        }

        // ── Results ──────────────────────────────────────────────────────
        AnimatedVisibility(
            visible = evalState.result != null && !evalState.isRunning,
            enter = fadeIn() + slideInVertically { it / 3 },
            exit = fadeOut()
        ) {
            evalState.result?.let { result ->
                Column(modifier = Modifier.padding(top = 20.dp)) {
                    // ── Primary Metrics Grid ─────────────────────────────
                    Text(
                        text = "CLASSIFICATION METRICS",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        letterSpacing = 1.5.sp
                    )
                    Spacer(modifier = Modifier.height(10.dp))

                    // Row 1: Accuracy, Precision, Recall, F1
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MetricCard("Accuracy", result.accuracyDisplay, PhishGuardBlue, Modifier.weight(1f))
                        MetricCard("Precision", result.precisionDisplay, AccentCyan, Modifier.weight(1f))
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MetricCard("Recall", result.recallDisplay, SafeGreen, Modifier.weight(1f))
                        MetricCard("F1 Score", result.f1Display, AccentPurple, Modifier.weight(1f))
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    // Row 2: ROC-AUC, FNR, FPR
                    Text(
                        text = "EXTENDED METRICS",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        letterSpacing = 1.5.sp
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MetricCard("ROC AUC", result.rocAucDisplay, PhishGuardBlue, Modifier.weight(1f))
                        MetricCard("FNR", result.fnrDisplay, DangerRed, Modifier.weight(1f))
                        MetricCard("FPR", result.fprDisplay, WarningAmber, Modifier.weight(1f))
                    }

                    Spacer(modifier = Modifier.height(20.dp))

                    // ── Confusion Matrix ─────────────────────────────────
                    Text(
                        text = "CONFUSION MATRIX",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        letterSpacing = 1.5.sp
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    ConfusionMatrixCard(result)

                    Spacer(modifier = Modifier.height(20.dp))

                    // ── Performance Stats ────────────────────────────────
                    Text(
                        text = "PERFORMANCE",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        letterSpacing = 1.5.sp
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    PerformanceCard(result)
                }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric Card — premium stat display
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun MetricCard(title: String, value: String, color: Color, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = value,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                color = color
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = title,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Confusion Matrix — visual 2×2 grid
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun ConfusionMatrixCard(result: CsvEvaluationResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // Header labels
            Row(modifier = Modifier.fillMaxWidth()) {
                Spacer(modifier = Modifier.weight(1f))
                Text(
                    text = "Predicted",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(2f),
                    textAlign = TextAlign.Center
                )
            }
            Row(modifier = Modifier.fillMaxWidth()) {
                Spacer(modifier = Modifier.weight(1f))
                Text(
                    text = "Benign (L)",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f),
                    textAlign = TextAlign.Center
                )
                Text(
                    text = "Phishing (M)",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f),
                    textAlign = TextAlign.Center
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Row 1: Actual Benign → TN, FP
            Row(
                modifier = Modifier.fillMaxWidth().height(72.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Row label
                Column(
                    modifier = Modifier.weight(1f),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text("Actual", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text("Benign", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                // TN cell (L→L) — correct → green
                ConfusionCell(
                    value = result.tn,
                    label = "TN (L→L)",
                    isCorrect = true,
                    modifier = Modifier.weight(1f)
                )
                Spacer(modifier = Modifier.width(4.dp))
                // FP cell (L→M) — error → red
                ConfusionCell(
                    value = result.fp,
                    label = "FP (L→M)",
                    isCorrect = false,
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Row 2: Actual Phishing → FN, TP
            Row(
                modifier = Modifier.fillMaxWidth().height(72.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text("Actual", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text("Phishing", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                // FN cell (M→L) — error → red
                ConfusionCell(
                    value = result.fn,
                    label = "FN (M→L)",
                    isCorrect = false,
                    modifier = Modifier.weight(1f)
                )
                Spacer(modifier = Modifier.width(4.dp))
                // TP cell (M→M) — correct → green
                ConfusionCell(
                    value = result.tp,
                    label = "TP (M→M)",
                    isCorrect = true,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
private fun ConfusionCell(value: Int, label: String, isCorrect: Boolean, modifier: Modifier) {
    val bgColor = if (isCorrect) SafeGreenSurface else DangerRedSurface
    val textColor = if (isCorrect) SafeGreen else DangerRed
    val borderColor = if (isCorrect) SafeGreen.copy(alpha = 0.3f) else DangerRed.copy(alpha = 0.3f)

    Box(
        modifier = modifier
            .fillMaxHeight()
            .clip(RoundedCornerShape(10.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(10.dp)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "%,d".format(value),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                color = textColor
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = textColor.copy(alpha = 0.7f),
                fontSize = 9.sp
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance Stats Card
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun PerformanceCard(result: CsvEvaluationResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            PerfRow("Total URLs", "%,d".format(result.totalUrls))
            PerfRow("Classified", "%,d".format(result.totalProcessed))
            PerfRow("Errors Skipped", "%,d".format(result.totalErrors))
            HorizontalDivider(
                modifier = Modifier.padding(vertical = 6.dp),
                color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
            )
            PerfRow("Elapsed Time", result.elapsedDisplay)
            PerfRow("Throughput", result.throughputDisplay)
            PerfRow("Threshold", "${"%.3f".format(result.threshold)}")
        }
    }
}

@Composable
private fun PerfRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Medium,
            fontFamily = FontFamily.Monospace,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
}
