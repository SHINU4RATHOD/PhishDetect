package com.phishguard.app.domain

import com.phishguard.app.data.PhishingUrlDetector

/**
 * Use-case layer: orchestrates a URL scan.
 * Provides clean separation between UI and data layer.
 */
class ScanUrlUseCase(private val detector: PhishingUrlDetector) {

    /**
     * Execute a scan for the given URL.
     * This should be called from a coroutine (IO dispatcher).
     *
     * @param url Raw URL string
     * @return DetectionResult with all fields populated
     */
    fun execute(url: String): DetectionResult {
        return detector.predict(url)
    }

    /**
     * Check if the detector is ready for inference.
     */
    val isReady: Boolean
        get() = detector.isReady
}
