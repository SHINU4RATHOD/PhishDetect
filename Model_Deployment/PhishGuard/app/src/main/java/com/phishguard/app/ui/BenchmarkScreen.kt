package com.phishguard.app.ui

import androidx.compose.animation.*
import androidx.compose.foundation.background
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
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.phishguard.app.domain.BenchmarkResult
import com.phishguard.app.ui.theme.*

@Composable
fun BenchmarkScreen(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    val benchState by viewModel.benchmarkState.collectAsState()
    val goldenState by viewModel.goldenTestState.collectAsState()

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
            text = "Performance Benchmark",
            style = MaterialTheme.typography.headlineLarge,
            color = MaterialTheme.colorScheme.onBackground
        )
        Text(
            text = "Warmup 5 + Timed 20 runs • Full pipeline",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(24.dp))

        // ── Run Benchmark Button ────────────────────────────────────────
        Button(
            onClick = { viewModel.runBenchmark() },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            enabled = uiState.isReady && !benchState.isRunning,
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = AccentPurple
            )
        ) {
            if (benchState.isRunning) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    color = androidx.compose.ui.graphics.Color.White,
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Running... ${"%.0f".format(benchState.progress * 100)}%")
            } else {
                Icon(Icons.Default.Speed, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Run Benchmark", fontWeight = FontWeight.SemiBold)
            }
        }

        // ── Progress Bar ─────────────────────────────────────────────────
        AnimatedVisibility(visible = benchState.isRunning) {
            Column(modifier = Modifier.padding(top = 12.dp)) {
                LinearProgressIndicator(
                    progress = { benchState.progress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp)),
                    color = AccentPurple,
                    trackColor = MaterialTheme.colorScheme.surfaceVariant
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // ── Benchmark Results ────────────────────────────────────────────
        AnimatedVisibility(
            visible = benchState.result != null,
            enter = fadeIn() + slideInVertically { it / 3 },
            exit = fadeOut()
        ) {
            benchState.result?.let { result ->
                BenchmarkResultsCard(result = result)
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // ── Golden Vector Parity Test ────────────────────────────────────
        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f))
        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Golden Vector Parity Test",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground
        )
        Text(
            text = "Token-ID match + probability tolerance (ε=0.001)",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(12.dp))

        OutlinedButton(
            onClick = { viewModel.runGoldenTest() },
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp),
            enabled = uiState.isReady && !goldenState.isRunning,
            shape = RoundedCornerShape(14.dp)
        ) {
            if (goldenState.isRunning) {
                CircularProgressIndicator(
                    modifier = Modifier.size(18.dp),
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Running tests...")
            } else {
                Icon(Icons.Default.Check, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Run Parity Test", fontWeight = FontWeight.SemiBold)
            }
        }

        // ── Golden Test Results ──────────────────────────────────────────
        AnimatedVisibility(
            visible = goldenState.summary.isNotEmpty() && !goldenState.isRunning,
            enter = fadeIn() + slideInVertically { it / 3 },
            exit = fadeOut()
        ) {
            Column(modifier = Modifier.padding(top = 12.dp)) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (goldenState.result?.allPassed == true)
                            SafeGreenSurface else DangerRedSurface
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = goldenState.summary,
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold,
                            color = if (goldenState.result?.allPassed == true)
                                SafeGreen else DangerRed
                        )

                        goldenState.result?.let { result ->
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Stage 1 (Tokenizer): ${result.stage1Passed}/${result.totalTests}",
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                text = "Stage 2 (Inference): ${result.stage2Passed}/${result.totalTests}",
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                                color = MaterialTheme.colorScheme.onSurface
                            )

                            // Show per-URL details for failures
                            result.results.filter { !it.stage1Pass || !it.stage2Pass }.forEach { testResult ->
                                Spacer(modifier = Modifier.height(8.dp))
                                HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "❌ ${testResult.url}",
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = FontWeight.SemiBold,
                                    color = DangerRed
                                )
                                testResult.stage1Error?.let {
                                    Text(
                                        text = "  Token: $it",
                                        style = MaterialTheme.typography.bodySmall,
                                        fontFamily = FontFamily.Monospace,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                                testResult.stage2Error?.let {
                                    Text(
                                        text = "  Prob: $it",
                                        style = MaterialTheme.typography.bodySmall,
                                        fontFamily = FontFamily.Monospace,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Not Ready State ──────────────────────────────────────────────
        if (!uiState.isReady && !uiState.isLoading) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = DangerRedSurface
                )
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.Error, contentDescription = null, tint = DangerRed)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Model not loaded. Cannot benchmark.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = DangerRed
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))
    }
}

@Composable
private fun BenchmarkResultsCard(result: BenchmarkResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            // Title
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Analytics,
                    contentDescription = null,
                    tint = AccentPurple,
                    modifier = Modifier.size(22.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Results",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ── Latency Stats ────────────────────────────────────────────
            Text(
                text = "END-TO-END LATENCY",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                LatencyStat("p50", "${"%.1f".format(result.p50TotalMs)} ms", AccentCyan)
                LatencyStat("p90", "${"%.1f".format(result.p90TotalMs)} ms", WarningAmber)
                LatencyStat("Mean", "${"%.1f".format(result.meanTotalMs)} ms", PhishGuardBlue)
            }

            Spacer(modifier = Modifier.height(16.dp))
            HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f))
            Spacer(modifier = Modifier.height(16.dp))

            // ── Breakdown ────────────────────────────────────────────────
            Text(
                text = "TIMING BREAKDOWN",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(8.dp))

            BenchmarkDetailRow("Tokenize (mean)", "${"%.2f".format(result.meanTokenizeMs)} ms")
            BenchmarkDetailRow("Inference (mean)", "${"%.2f".format(result.meanInferenceMs)} ms")
            BenchmarkDetailRow("Total (mean)", "${"%.2f".format(result.meanTotalMs)} ms")
            BenchmarkDetailRow("Min", "${"%.2f".format(result.minTotalMs)} ms")
            BenchmarkDetailRow("Max", "${"%.2f".format(result.maxTotalMs)} ms")

            Spacer(modifier = Modifier.height(16.dp))
            HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f))
            Spacer(modifier = Modifier.height(16.dp))

            // ── Device Info ──────────────────────────────────────────────
            Text(
                text = "ENVIRONMENT",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(8.dp))

            BenchmarkDetailRow("Execution Provider", result.executionProvider)
            BenchmarkDetailRow("Timed Runs", "${result.totalRuns}")
            BenchmarkDetailRow("Warmup Runs", "${result.warmupRuns}")

            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = result.deviceInfo,
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun LatencyStat(
    label: String,
    value: String,
    color: androidx.compose.ui.graphics.Color
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = value,
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            color = color
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun BenchmarkDetailRow(label: String, value: String) {
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
