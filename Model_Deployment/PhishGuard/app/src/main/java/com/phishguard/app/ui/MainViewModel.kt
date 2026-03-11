package com.phishguard.app.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.phishguard.app.benchmark.BenchmarkRunner
import com.phishguard.app.core.PhishGuardConfig
import com.phishguard.app.data.CsvEvaluator
import com.phishguard.app.data.PhishingUrlDetector
import com.phishguard.app.domain.BenchmarkResult
import com.phishguard.app.domain.CsvEvaluationResult
import com.phishguard.app.domain.DetectionResult
import com.phishguard.app.domain.ScanUrlUseCase
import com.phishguard.app.test.GoldenVectorTest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Main ViewModel managing detector lifecycle, scan history, and benchmark state.
 * Uses AndroidViewModel for Application context (needed by ONNX/tokenizer).
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    // ── State ────────────────────────────────────────────────────────────────
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _benchmarkState = MutableStateFlow(BenchmarkState())
    val benchmarkState: StateFlow<BenchmarkState> = _benchmarkState.asStateFlow()

    private val _goldenTestState = MutableStateFlow(GoldenTestState())
    val goldenTestState: StateFlow<GoldenTestState> = _goldenTestState.asStateFlow()

    private val _evaluationState = MutableStateFlow(EvaluationState())
    val evaluationState: StateFlow<EvaluationState> = _evaluationState.asStateFlow()

    // ── Dependencies ─────────────────────────────────────────────────────────
    private var detector: PhishingUrlDetector? = null
    private var scanUseCase: ScanUrlUseCase? = null
    private var benchmarkRunner: BenchmarkRunner? = null

    init {
        initializeDetector()
    }

    /**
     * Initialize detector on IO dispatcher (model loading is heavy).
     */
    private fun initializeDetector() {
        _uiState.value = _uiState.value.copy(isLoading = true, statusMessage = "Loading model...")

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val det = PhishingUrlDetector(getApplication())
                val success = det.initialize(enableNnapi = true)

                if (success) {
                    detector = det
                    scanUseCase = ScanUrlUseCase(det)
                    benchmarkRunner = BenchmarkRunner(det)

                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        isReady = true,
                        statusMessage = "Ready • ${det.executionProvider}",
                        executionProvider = det.executionProvider
                    )
                } else {
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        isReady = false,
                        statusMessage = "Model initialization failed"
                    )
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    isReady = false,
                    statusMessage = "Error: ${e.message}"
                )
            }
        }
    }

    /**
     * Scan a URL on IO dispatcher.
     */
    fun scanUrl(url: String) {
        if (!(_uiState.value.isReady) || _uiState.value.isScanning) return

        _uiState.value = _uiState.value.copy(
            isScanning = true,
            currentResult = null,
            statusMessage = "Scanning..."
        )

        viewModelScope.launch(Dispatchers.IO) {
            val result = scanUseCase?.execute(url) ?: DetectionResult.error("Detector not ready")

            _uiState.value = _uiState.value.copy(
                isScanning = false,
                currentResult = result,
                scanHistory = listOf(result) + _uiState.value.scanHistory.take(49), // Keep last 50
                statusMessage = if (result.isSuccess) "Scan complete" else "Scan failed"
            )
        }
    }

    /**
     * Run benchmark on IO dispatcher.
     */
    fun runBenchmark() {
        if (!(_uiState.value.isReady) || _benchmarkState.value.isRunning) return

        _benchmarkState.value = _benchmarkState.value.copy(
            isRunning = true,
            progress = 0f,
            result = null
        )

        viewModelScope.launch(Dispatchers.IO) {
            val result = benchmarkRunner?.run(
                onProgress = { progress ->
                    _benchmarkState.value = _benchmarkState.value.copy(progress = progress)
                }
            )

            _benchmarkState.value = _benchmarkState.value.copy(
                isRunning = false,
                progress = 1f,
                result = result
            )
        }
    }

    /**
     * Clear scan history.
     */
    fun clearHistory() {
        _uiState.value = _uiState.value.copy(scanHistory = emptyList())
    }

    /**
     * Run golden vector parity tests on IO dispatcher.
     */
    fun runGoldenTest() {
        val det = detector ?: return
        if (_goldenTestState.value.isRunning) return

        val tok = det.tokenizer ?: return
        val model = det.modelLoader ?: return

        _goldenTestState.value = _goldenTestState.value.copy(
            isRunning = true,
            result = null,
            summary = "Running golden tests..."
        )

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val test = GoldenVectorTest(getApplication(), tok, model)
                val result = test.runTests()

                _goldenTestState.value = _goldenTestState.value.copy(
                    isRunning = false,
                    result = result,
                    summary = if (result.allPassed)
                        "✅ All ${result.totalTests} tests passed"
                    else
                        "❌ Stage1: ${result.stage1Passed}/${result.totalTests}, Stage2: ${result.stage2Passed}/${result.totalTests}"
                )
            } catch (e: Exception) {
                _goldenTestState.value = _goldenTestState.value.copy(
                    isRunning = false,
                    summary = "Error: ${e.message}"
                )
            }
        }
    }

    /**
     * Handle CSV file selection — pre-scan for row count and column validation.
     */
    fun onCsvSelected(uri: Uri) {
        val det = detector ?: return
        viewModelScope.launch(Dispatchers.IO) {
            val evaluator = CsvEvaluator(getApplication(), det)
            val info = evaluator.scanCsv(uri)

            _evaluationState.value = _evaluationState.value.copy(
                csvUri = uri,
                csvFileName = info.fileName,
                csvTotalRows = info.totalRows,
                error = if (!info.hasRequiredColumns) info.error else null,
                result = null
            )
        }
    }

    /**
     * Run CSV evaluation on IO dispatcher.
     */
    fun runEvaluation() {
        val det = detector ?: return
        val uri = _evaluationState.value.csvUri ?: return
        if (_evaluationState.value.isRunning) return

        _evaluationState.value = _evaluationState.value.copy(
            isRunning = true,
            progress = 0f,
            processed = 0,
            result = null,
            error = null
        )

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val evaluator = CsvEvaluator(getApplication(), det)
                val result = evaluator.evaluate(uri) { processed, total ->
                    _evaluationState.value = _evaluationState.value.copy(
                        processed = processed,
                        progress = if (total > 0) processed.toFloat() / total else 0f
                    )
                }

                _evaluationState.value = _evaluationState.value.copy(
                    isRunning = false,
                    progress = 1f,
                    result = result
                )
            } catch (e: Exception) {
                _evaluationState.value = _evaluationState.value.copy(
                    isRunning = false,
                    error = "Evaluation failed: ${e.message}"
                )
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        detector?.close()
    }
}

/**
 * UI state for the main scan screen.
 */
data class UiState(
    val isLoading: Boolean = true,
    val isReady: Boolean = false,
    val isScanning: Boolean = false,
    val statusMessage: String = "Initializing...",
    val executionProvider: String = "—",
    val currentResult: DetectionResult? = null,
    val scanHistory: List<DetectionResult> = emptyList()
)

/**
 * State for the benchmark screen.
 */
data class BenchmarkState(
    val isRunning: Boolean = false,
    val progress: Float = 0f,
    val result: BenchmarkResult? = null
)

/**
 * State for the golden vector test.
 */
data class GoldenTestState(
    val isRunning: Boolean = false,
    val result: GoldenVectorTest.TestSuiteResult? = null,
    val summary: String = ""
)

/**
 * State for CSV evaluation.
 */
data class EvaluationState(
    val isRunning: Boolean = false,
    val progress: Float = 0f,
    val processed: Int = 0,
    val csvUri: Uri? = null,
    val csvFileName: String = "",
    val csvTotalRows: Int = 0,
    val result: CsvEvaluationResult? = null,
    val error: String? = null
)
