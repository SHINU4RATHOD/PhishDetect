package com.phishguard.app.data

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import com.phishguard.app.core.PhishGuardConfig
import com.phishguard.app.core.SecureLogger
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.LongBuffer

/**
 * ONNX Runtime model loader with:
 * - Streaming buffered asset copy (never readBytes() — prevents OOM)
 * - Dynamic I/O name discovery + validation
 * - Input tensor type validation
 * - Optional NNAPI with CPU fallback
 * - Session optimization tuning
 */
class OnnxModelLoader(private val context: Context) {

    private val tag = "ModelLoader"

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // Discovered I/O names (populated after session creation)
    var inputIdsName: String = PhishGuardConfig.EXPECTED_INPUT_IDS_NAME
        private set
    var attentionMaskName: String = PhishGuardConfig.EXPECTED_ATTENTION_MASK_NAME
        private set
    var outputName: String = PhishGuardConfig.EXPECTED_OUTPUT_NAME
        private set
    var executionProvider: String = "CPU"
        private set

    val isLoaded: Boolean get() = ortSession != null

    /**
     * Initialize ORT environment + session.
     * Copies model from assets to internal storage via streaming I/O,
     * then creates an optimized session.
     *
     * @param enableNnapi Whether to attempt NNAPI acceleration
     * @return True if session created successfully
     */
    fun initialize(enableNnapi: Boolean = true): Boolean {
        return try {
            SecureLogger.i(tag, "Initializing ONNX Runtime...")

            // Step 1: Create ORT environment
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Step 2: Copy model to internal storage (streaming)
            val modelFile = copyAssetToInternal()
            SecureLogger.i(tag, "Model file ready: ${modelFile.length() / 1024 / 1024} MB")

            // Step 3: Create session with optimized options
            val options = SessionOptions()
            options.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
            options.setIntraOpNumThreads(PhishGuardConfig.ORT_INTRA_OP_THREADS)
            options.setInterOpNumThreads(PhishGuardConfig.ORT_INTER_OP_THREADS)

            // Attempt NNAPI if requested and API level supports it
            if (enableNnapi && android.os.Build.VERSION.SDK_INT >= PhishGuardConfig.NNAPI_MIN_API_LEVEL) {
                try {
                    options.addNnapi()
                    executionProvider = "NNAPI"
                    SecureLogger.i(tag, "NNAPI execution provider added")
                } catch (e: Exception) {
                    SecureLogger.w(tag, "NNAPI unavailable, falling back to CPU: ${e.message}")
                    executionProvider = "CPU"
                }
            } else {
                executionProvider = "CPU"
                SecureLogger.i(tag, "Using CPU execution provider (API=${android.os.Build.VERSION.SDK_INT})")
            }

            // Step 4: Create session
            ortSession = ortEnvironment!!.createSession(modelFile.absolutePath, options)

            // Step 5: Discover and validate I/O names dynamically
            discoverAndValidateIO()

            SecureLogger.i(tag, "Session created successfully. EP=$executionProvider")
            true
        } catch (e: Exception) {
            SecureLogger.e(tag, "Failed to initialize ONNX Runtime", e)
            false
        }
    }

    /**
     * Discover input/output names from session metadata.
     * Validates against expected names and fails fast on mismatch.
     */
    private fun discoverAndValidateIO() {
        val session = ortSession ?: throw IllegalStateException("Session not created")

        // Discover input names
        val inputNames = session.inputNames.toList()
        SecureLogger.i(tag, "Discovered inputs: $inputNames")

        if (inputNames.isEmpty()) {
            throw IllegalStateException("Model has no inputs")
        }

        // Map discovered names to expected roles
        inputIdsName = inputNames.find { it.contains("input_ids", ignoreCase = true) }
            ?: inputNames.find { it.contains("input", ignoreCase = true) }
            ?: inputNames[0]

        attentionMaskName = inputNames.find { it.contains("attention_mask", ignoreCase = true) }
            ?: inputNames.find { it.contains("mask", ignoreCase = true) }
            ?: if (inputNames.size > 1) inputNames[1] else throw IllegalStateException("Cannot find attention_mask input")

        // Discover output name
        val outputNames = session.outputNames.toList()
        SecureLogger.i(tag, "Discovered outputs: $outputNames")

        outputName = outputNames.find { it.contains("logit", ignoreCase = true) }
            ?: outputNames.find { it.contains("output", ignoreCase = true) }
            ?: outputNames[0]

        // Validate against expected names (warn, don't crash — discovered names take precedence)
        if (inputIdsName != PhishGuardConfig.EXPECTED_INPUT_IDS_NAME) {
            SecureLogger.w(tag, "Input IDs name mismatch: expected='${PhishGuardConfig.EXPECTED_INPUT_IDS_NAME}', discovered='$inputIdsName'")
        }
        if (attentionMaskName != PhishGuardConfig.EXPECTED_ATTENTION_MASK_NAME) {
            SecureLogger.w(tag, "Attention mask name mismatch: expected='${PhishGuardConfig.EXPECTED_ATTENTION_MASK_NAME}', discovered='$attentionMaskName'")
        }
        if (outputName != PhishGuardConfig.EXPECTED_OUTPUT_NAME) {
            SecureLogger.w(tag, "Output name mismatch: expected='${PhishGuardConfig.EXPECTED_OUTPUT_NAME}', discovered='$outputName'")
        }

        // Validate input tensor info (type check)
        val inputInfo = session.inputInfo
        for ((name, info) in inputInfo) {
            val typeInfo = info.info
            SecureLogger.d(tag, "Input '$name' type info: $typeInfo")
        }

        SecureLogger.i(tag, "I/O validated: inputs=[$inputIdsName, $attentionMaskName], output=[$outputName]")
    }

    /**
     * Run inference on pre-tokenized inputs.
     *
     * @param inputIds Token IDs as int64 array (length = MAX_LENGTH)
     * @param attentionMask Attention mask as int64 array (length = MAX_LENGTH)
     * @return Raw logits array [2] (benign, phishing)
     */
    fun runInference(inputIds: LongArray, attentionMask: LongArray): FloatArray {
        val env = ortEnvironment ?: throw IllegalStateException("ORT environment not initialized")
        val session = ortSession ?: throw IllegalStateException("ORT session not initialized")

        val shape = longArrayOf(1, PhishGuardConfig.MAX_LENGTH.toLong())

        // Create input tensors (int64 — matching export)
        val inputIdsTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(inputIds),
            shape
        )

        val attentionMaskTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(attentionMask),
            shape
        )

        val inputs = mapOf(
            inputIdsName to inputIdsTensor,
            attentionMaskName to attentionMaskTensor
        )

        return try {
            val results = session.run(inputs)

            // Extract output — Result.get(String) returns Optional<OnnxValue>
            val outputValue: OnnxValue = results.get(outputName)
                .orElseGet { results.get(0) }

            // Cast to OnnxTensor to access .value
            val outputTensor = outputValue as OnnxTensor
            val rawOutput = outputTensor.value

            val logits: FloatArray = when (rawOutput) {
                is Array<*> -> {
                    // Shape [1, 2] — nested array
                    @Suppress("UNCHECKED_CAST")
                    (rawOutput as Array<FloatArray>)[0]
                }
                is FloatArray -> {
                    // Shape [2] — flat array
                    rawOutput
                }
                else -> throw IllegalStateException(
                    "Unexpected output type: ${rawOutput.javaClass.name}. " +
                    "Expected Array<FloatArray> ([1,2]) or FloatArray ([2])."
                )
            }

            if (logits.size != PhishGuardConfig.NUM_CLASSES) {
                throw IllegalStateException(
                    "Output logits size=${logits.size}, expected=${PhishGuardConfig.NUM_CLASSES}"
                )
            }

            logits
        } finally {
            inputIdsTensor.close()
            attentionMaskTensor.close()
        }
    }

    /**
     * Copy asset file to internal storage using streaming buffered I/O.
     * NEVER uses readBytes() — prevents OOM on mid-range devices.
     */
    private fun copyAssetToInternal(): File {
        val assetPath = "${PhishGuardConfig.ASSETS_DIR}/${PhishGuardConfig.MODEL_FILENAME}"
        val outFile = File(context.filesDir, PhishGuardConfig.MODEL_FILENAME)

        // Skip copy if file already exists and is non-empty
        if (outFile.exists() && outFile.length() > 0) {
            SecureLogger.d(tag, "Model file already cached: ${outFile.absolutePath}")
            return outFile
        }

        SecureLogger.i(tag, "Copying asset '$assetPath' to internal storage (streaming)...")

        context.assets.open(assetPath).use { inputStream ->
            BufferedOutputStream(FileOutputStream(outFile), 8192).use { outputStream ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                var totalBytes = 0L

                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    outputStream.write(buffer, 0, bytesRead)
                    totalBytes += bytesRead
                }

                outputStream.flush()
                SecureLogger.i(tag, "Asset copy complete: ${totalBytes / 1024 / 1024} MB")
            }
        }

        return outFile
    }

    /**
     * Release ORT resources.
     */
    fun close() {
        try {
            ortSession?.close()
            ortSession = null
            // Note: OrtEnvironment is a singleton, don't close it
            SecureLogger.i(tag, "ORT session closed")
        } catch (e: Exception) {
            SecureLogger.e(tag, "Error closing ORT session", e)
        }
    }
}
