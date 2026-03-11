package com.phishguard.app.core

import android.util.Log
import java.security.MessageDigest

/**
 * Secure logging utility that hashes sensitive data (URLs) before logging.
 * Prevents PII/URL leakage in production logs while maintaining debuggability.
 */
object SecureLogger {

    private const val TAG = "PhishGuard"
    private val sha256 = MessageDigest.getInstance("SHA-256")

    /**
     * Log at DEBUG level with URL hashed.
     */
    fun d(subtag: String, message: String, url: String? = null) {
        val safeMsg = if (url != null) {
            "$message [url_hash=${hashUrl(url)}]"
        } else {
            message
        }
        Log.d("$TAG/$subtag", safeMsg)
    }

    /**
     * Log at INFO level with URL hashed.
     */
    fun i(subtag: String, message: String, url: String? = null) {
        val safeMsg = if (url != null) {
            "$message [url_hash=${hashUrl(url)}]"
        } else {
            message
        }
        Log.i("$TAG/$subtag", safeMsg)
    }

    /**
     * Log at WARN level with URL hashed.
     */
    fun w(subtag: String, message: String, url: String? = null) {
        val safeMsg = if (url != null) {
            "$message [url_hash=${hashUrl(url)}]"
        } else {
            message
        }
        Log.w("$TAG/$subtag", safeMsg)
    }

    /**
     * Log at ERROR level with URL hashed.
     */
    fun e(subtag: String, message: String, throwable: Throwable? = null, url: String? = null) {
        val safeMsg = if (url != null) {
            "$message [url_hash=${hashUrl(url)}]"
        } else {
            message
        }
        if (throwable != null) {
            Log.e("$TAG/$subtag", safeMsg, throwable)
        } else {
            Log.e("$TAG/$subtag", safeMsg)
        }
    }

    /**
     * Hash a URL using SHA-256, returning first 12 hex characters.
     * Enough for debugging correlation without exposing the actual URL.
     */
    fun hashUrl(url: String): String {
        return try {
            val digest = sha256.digest(url.toByteArray(Charsets.UTF_8))
            digest.take(6).joinToString("") { "%02x".format(it) }
        } catch (_: Exception) {
            "hash_error"
        }
    }
}
