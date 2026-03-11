package com.phishguard.app.ui

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.phishguard.app.core.PhishGuardConfig
import com.phishguard.app.domain.DetectionResult
import com.phishguard.app.ui.theme.*

@Composable
fun ScanScreen(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    var urlInput by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp)
    ) {
        Spacer(modifier = Modifier.height(16.dp))

        // ── Header ───────────────────────────────────────────────────────
        HeaderSection(uiState = uiState)

        Spacer(modifier = Modifier.height(24.dp))

        // ── URL Input ────────────────────────────────────────────────────
        UrlInputSection(
            urlInput = urlInput,
            onUrlChange = { urlInput = it },
            onScan = { viewModel.scanUrl(urlInput) },
            isScanning = uiState.isScanning,
            isReady = uiState.isReady
        )

        Spacer(modifier = Modifier.height(16.dp))

        // ── Sample URL Chips ─────────────────────────────────────────────
        SampleUrlChips(
            onUrlSelected = { url ->
                urlInput = url
                viewModel.scanUrl(url)
            },
            isReady = uiState.isReady && !uiState.isScanning
        )

        Spacer(modifier = Modifier.height(24.dp))

        // ── Result Card ──────────────────────────────────────────────────
        AnimatedVisibility(
            visible = uiState.currentResult != null,
            enter = fadeIn(animationSpec = tween(400)) +
                    slideInVertically(animationSpec = tween(400)) { it / 3 },
            exit = fadeOut(animationSpec = tween(200))
        ) {
            uiState.currentResult?.let { result ->
                ResultCard(result = result)
            }
        }

        // ── Loading Indicator ────────────────────────────────────────────
        AnimatedVisibility(visible = uiState.isScanning) {
            ScanningIndicator()
        }

        Spacer(modifier = Modifier.height(32.dp))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Header
// ═════════════════════════════════════════════════════════════════════════════

@Composable
private fun HeaderSection(uiState: UiState) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Shield icon with gradient background
        Box(
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(
                    Brush.linearGradient(
                        colors = listOf(PhishGuardBlue, AccentCyan)
                    )
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.Shield,
                contentDescription = "PhishGuard",
                tint = Color.White,
                modifier = Modifier.size(28.dp)
            )
        }

        Spacer(modifier = Modifier.width(14.dp))

        Column {
            Text(
                text = "PhishGuard",
                style = MaterialTheme.typography.headlineLarge,
                color = MaterialTheme.colorScheme.onBackground
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(
                            if (uiState.isReady) SafeGreen
                            else if (uiState.isLoading) WarningAmber
                            else DangerRed
                        )
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = uiState.statusMessage,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// URL Input
// ═════════════════════════════════════════════════════════════════════════════

@Composable
private fun UrlInputSection(
    urlInput: String,
    onUrlChange: (String) -> Unit,
    onScan: () -> Unit,
    isScanning: Boolean,
    isReady: Boolean
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Enter URL to scan",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = urlInput,
                onValueChange = onUrlChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("https://example.com") },
                leadingIcon = {
                    Icon(Icons.Outlined.Link, contentDescription = null)
                },
                trailingIcon = {
                    if (urlInput.isNotEmpty()) {
                        IconButton(onClick = { onUrlChange("") }) {
                            Icon(Icons.Default.Clear, contentDescription = "Clear")
                        }
                    }
                },
                singleLine = true,
                shape = RoundedCornerShape(12.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = PhishGuardBlue,
                    cursorColor = PhishGuardBlue
                )
            )

            Spacer(modifier = Modifier.height(12.dp))

            Button(
                onClick = onScan,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                enabled = urlInput.isNotBlank() && isReady && !isScanning,
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = PhishGuardBlue,
                    disabledContainerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                if (isScanning) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = Color.White,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Scanning...")
                } else {
                    Icon(Icons.Default.Search, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Scan URL", fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Sample URL Chips
// ═════════════════════════════════════════════════════════════════════════════

@Composable
private fun SampleUrlChips(onUrlSelected: (String) -> Unit, isReady: Boolean) {
    Column {
        Text(
            text = "Quick Scan",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(8.dp))
        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(PhishGuardConfig.SAMPLE_URLS) { url ->
                val displayText = url
                    .removePrefix("https://")
                    .removePrefix("http://")
                    .take(28)
                    .let { if (it.length == 28) "$it..." else it }

                SuggestionChip(
                    onClick = { if (isReady) onUrlSelected(url) },
                    label = {
                        Text(
                            text = displayText,
                            style = MaterialTheme.typography.bodySmall,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    },
                    enabled = isReady,
                    shape = RoundedCornerShape(20.dp)
                )
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Result Card
// ═════════════════════════════════════════════════════════════════════════════

@Composable
private fun ResultCard(result: DetectionResult) {
    var showDetails by remember { mutableStateOf(false) }

    val verdictColor = if (result.isPhishing) DangerRed else SafeGreen
    val verdictSurfaceColor = if (result.isPhishing) DangerRedSurface else SafeGreenSurface
    val verdictIcon = if (result.isPhishing) Icons.Default.Warning else Icons.Default.VerifiedUser

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(
                elevation = 8.dp,
                shape = RoundedCornerShape(20.dp),
                spotColor = verdictColor.copy(alpha = 0.3f)
            ),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            // Error state
            if (result.isError) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.Error,
                        contentDescription = null,
                        tint = DangerRed,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = result.error ?: "Unknown error",
                        style = MaterialTheme.typography.bodyMedium,
                        color = DangerRed
                    )
                }
                return@Card
            }

            // ── Verdict Banner ───────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(verdictSurfaceColor)
                    .border(1.dp, verdictColor.copy(alpha = 0.3f), RoundedCornerShape(14.dp))
                    .padding(16.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = verdictIcon,
                        contentDescription = null,
                        tint = verdictColor,
                        modifier = Modifier.size(36.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = result.label,
                            style = MaterialTheme.typography.headlineMedium,
                            fontWeight = FontWeight.Bold,
                            color = verdictColor
                        )
                        Text(
                            text = result.displayUrl,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ── Metrics Grid ─────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                MetricItem(
                    label = "Probability",
                    value = result.phishingPercentage,
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    label = "Threshold",
                    value = "${"%.0f".format(result.threshold * 100)}%",
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    label = "Latency",
                    value = result.formattedLatency,
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    label = "Provider",
                    value = result.executionProvider,
                    modifier = Modifier.weight(1f)
                )
            }

            // ── Normalization Warnings ────────────────────────────────────
            if (result.normalizationWarnings.isNotEmpty()) {
                Spacer(modifier = Modifier.height(12.dp))
                result.normalizationWarnings.forEach { warning ->
                    Row(
                        modifier = Modifier.padding(vertical = 2.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Outlined.Info,
                            contentDescription = null,
                            tint = WarningAmber,
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = warning,
                            style = MaterialTheme.typography.bodySmall,
                            color = WarningAmber
                        )
                    }
                }
            }

            // ── Expandable Details ───────────────────────────────────────
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .clickable { showDetails = !showDetails }
                    .padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                Text(
                    text = if (showDetails) "Hide Details" else "Show Details",
                    style = MaterialTheme.typography.labelMedium,
                    color = PhishGuardBlue
                )
                Spacer(modifier = Modifier.width(4.dp))
                Icon(
                    if (showDetails) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                    contentDescription = null,
                    tint = PhishGuardBlue,
                    modifier = Modifier.size(18.dp)
                )
            }

            AnimatedVisibility(visible = showDetails) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(MaterialTheme.colorScheme.surfaceVariant)
                        .padding(12.dp)
                ) {
                    DetailRow("Token Count", "${result.tokenCount}")
                    DetailRow("Max Length", "${result.maxLength}")
                    DetailRow("Tokenize Time", "${"%.2f".format(result.tokenizeTimeMs)} ms")
                    DetailRow("Inference Time", "${"%.2f".format(result.inferenceTimeMs)} ms")
                    DetailRow("Model", result.modelFileName)
                    DetailRow("P(benign)", "${"%.4f".format(result.benignProbability)}")
                    DetailRow("P(phishing)", "${"%.4f".format(result.phishingProbability)}")
                    if (result.containsPunycode) {
                        DetailRow("Punycode", "⚠️ Detected")
                    }
                    if (result.containsSuspiciousUnicode) {
                        DetailRow("Unicode Risk", "⚠️ Confusable chars detected")
                    }
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Helper Composables
// ═════════════════════════════════════════════════════════════════════════════

@Composable
private fun MetricItem(label: String, value: String, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun DetailRow(label: String, value: String) {
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

@Composable
private fun ScanningIndicator() {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(24.dp),
                color = PhishGuardBlue,
                strokeWidth = 3.dp
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = "Analyzing URL...",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
