package com.phishguard.app.core

import java.net.IDN
import java.text.Normalizer

/**
 * Cybersecurity-grade URL normalizer.
 *
 * Applies:
 * 1. Unicode NFKC normalization
 * 2. Control character stripping
 * 3. Whitespace collapsing
 * 4. Length capping
 * 5. Punycode/IDN detection and flagging
 * 6. Unicode confusable detection (basic)
 */
object UrlNormalizer {

    data class NormalizationResult(
        val normalizedUrl: String,
        val wasTruncated: Boolean,
        val containsPunycode: Boolean,
        val containsSuspiciousUnicode: Boolean,
        val warnings: List<String>
    )

    /**
     * Normalize a URL for safe, consistent tokenization.
     * This MUST match the normalization applied in Python golden vector generation.
     */
    fun normalize(url: String): NormalizationResult {
        val warnings = mutableListOf<String>()

        if (url.isBlank()) {
            return NormalizationResult(
                normalizedUrl = "",
                wasTruncated = false,
                containsPunycode = false,
                containsSuspiciousUnicode = false,
                warnings = listOf("Empty or blank URL")
            )
        }

        // Step 1: Unicode NFKC normalization
        // Converts compatibility characters to canonical form
        var normalized = Normalizer.normalize(url.trim(), Normalizer.Form.NFKC)

        // Step 2: Strip control characters (U+0000–U+001F, U+007F, U+0080–U+009F)
        // Keep standard printable ASCII + valid URL chars
        val beforeControl = normalized
        normalized = normalized.replace(Regex("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F\\x80-\\x9F]"), "")
        if (normalized.length != beforeControl.length) {
            warnings.add("Stripped ${beforeControl.length - normalized.length} control character(s)")
        }

        // Step 3: Collapse whitespace to single spaces, then remove all whitespace
        // URLs should not contain whitespace
        normalized = normalized.replace(Regex("\\s+"), " ").trim()

        // Step 4: Check for punycode / IDN (Internationalized Domain Names)
        val containsPunycode = detectPunycode(normalized)
        if (containsPunycode) {
            warnings.add("URL contains punycode/IDN — potential homograph attack vector")
            // Attempt to decode punycode for display
            normalized = decodePunycodeInUrl(normalized)
        }

        // Step 5: Detect suspicious Unicode characters (confusables)
        val containsSuspicious = detectSuspiciousUnicode(normalized)
        if (containsSuspicious) {
            warnings.add("URL contains Unicode characters that may be confusable with ASCII")
        }

        // Step 6: Cap length
        val wasTruncated = normalized.length > PhishGuardConfig.MAX_URL_LENGTH
        if (wasTruncated) {
            normalized = normalized.take(PhishGuardConfig.MAX_URL_LENGTH)
            warnings.add("URL truncated from ${url.length} to ${PhishGuardConfig.MAX_URL_LENGTH} characters")
        }

        return NormalizationResult(
            normalizedUrl = normalized,
            wasTruncated = wasTruncated,
            containsPunycode = containsPunycode,
            containsSuspiciousUnicode = containsSuspicious,
            warnings = warnings
        )
    }

    /**
     * Detect punycode patterns (xn--) in URL hostnames.
     */
    private fun detectPunycode(url: String): Boolean {
        return url.contains("xn--", ignoreCase = true)
    }

    /**
     * Attempt to decode punycode hostnames in a URL.
     */
    private fun decodePunycodeInUrl(url: String): String {
        return try {
            // Extract hostname and attempt IDN decode
            val regex = Regex("(https?://)?([^/]+)(.*)")
            val match = regex.find(url) ?: return url
            val scheme = match.groupValues[1]
            val host = match.groupValues[2]
            val path = match.groupValues[3]
            val decodedHost = IDN.toUnicode(host)
            "$scheme$decodedHost$path"
        } catch (_: Exception) {
            url  // Return original if decode fails
        }
    }

    /**
     * Detect Unicode characters commonly used in homograph attacks.
     * Checks for Cyrillic, Greek, and other scripts that contain
     * characters visually similar to Latin letters.
     */
    private fun detectSuspiciousUnicode(url: String): Boolean {
        for (char in url) {
            val block = Character.UnicodeBlock.of(char)
            if (block != null && block in SUSPICIOUS_UNICODE_BLOCKS) {
                return true
            }
        }
        return false
    }

    private val SUSPICIOUS_UNICODE_BLOCKS = setOf(
        Character.UnicodeBlock.CYRILLIC,
        Character.UnicodeBlock.CYRILLIC_SUPPLEMENTARY,
        Character.UnicodeBlock.GREEK,
        Character.UnicodeBlock.GREEK_EXTENDED,
        Character.UnicodeBlock.ARMENIAN,
        Character.UnicodeBlock.CHEROKEE,
        Character.UnicodeBlock.MYANMAR,
        // Fullwidth forms (e.g., ｇｏｏｇｌｅ.ｃｏｍ)
        Character.UnicodeBlock.HALFWIDTH_AND_FULLWIDTH_FORMS
    )
}
