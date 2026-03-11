package com.phishguard.app.data

import android.content.Context
import com.phishguard.app.core.PhishGuardConfig
import com.phishguard.app.core.SecureLogger
import java.io.BufferedReader
import java.io.InputStreamReader
import java.text.Normalizer

/**
 * Production-grade BERT WordPiece tokenizer implemented in Kotlin.
 *
 * Tokenization order (matches HuggingFace BasicTokenizer exactly):
 * 1. cleanText       — strip control chars, collapse whitespace
 * 2. tokenizeChineseChars — add spaces around CJK characters
 * 3. lowercase
 * 4. stripAccents    — NFD → remove combining marks
 * 5. splitOnPunctuation  — Unicode category-based punctuation splitting
 * 6. Whitespace tokenize → WordPiece greedy with ##
 * 7. Wrap: [CLS] + tokens + [SEP]
 * 8. Truncate (preserving [CLS] and [SEP]), then pad to MAX_LEN
 * 9. Generate attention mask
 *
 * @param context Android context for accessing vocab.txt from assets
 */
class BertWordPieceTokenizer(context: Context) {

    private val tag = "Tokenizer"
    private val vocab: Map<String, Int>        // token → id
    private val reverseVocab: Map<Int, String>  // id → token (for debug)
    private val maxLen = PhishGuardConfig.MAX_LENGTH

    init {
        val (v, rv) = loadVocab(context)
        vocab = v
        reverseVocab = rv
        SecureLogger.i(tag, "Vocabulary loaded: ${vocab.size} tokens, maxLen=$maxLen")
    }

    /**
     * Load vocabulary from assets/phishing/vocab.txt.
     * Each line is one token, line number is the token ID.
     */
    private fun loadVocab(context: Context): Pair<Map<String, Int>, Map<Int, String>> {
        val vocabMap = LinkedHashMap<String, Int>(32000)
        val reverseMap = HashMap<Int, String>(32000)

        context.assets.open("${PhishGuardConfig.ASSETS_DIR}/${PhishGuardConfig.VOCAB_FILENAME}").use { stream ->
            BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
                var index = 0
                reader.forEachLine { line ->
                    val token = line.trim()
                    if (token.isNotEmpty()) {
                        vocabMap[token] = index
                        reverseMap[index] = token
                        index++
                    }
                }
            }
        }

        SecureLogger.d(tag, "Loaded ${vocabMap.size} vocab entries")
        return Pair(vocabMap, reverseMap)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Public API
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Tokenize result with all tensors needed for ONNX inference.
     */
    data class TokenizeResult(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenCount: Int,       // Actual tokens (before padding)
        val tokens: List<String>   // Token strings for debug
    )

    /**
     * Full tokenization pipeline: text → input_ids + attention_mask (int64).
     *
     * @param text Input URL or text string (already normalized via UrlNormalizer)
     * @return TokenizeResult with int64 arrays ready for ORT inference
     */
    fun tokenize(text: String): TokenizeResult {
        // Step 1–5: BERT BasicTokenizer pipeline
        val basicTokens = basicTokenize(text)

        // Step 6: WordPiece sub-tokenization
        val wordPieceTokens = mutableListOf<String>()
        for (token in basicTokens) {
            wordPieceTokens.addAll(wordPieceTokenize(token))
        }

        // Step 7: Add [CLS] and [SEP]
        val allTokens = mutableListOf("[CLS]")
        allTokens.addAll(wordPieceTokens)
        allTokens.add("[SEP]")

        // Step 8: Truncate (preserving [CLS] at start and [SEP] at end)
        val truncatedTokens = if (allTokens.size > maxLen) {
            val result = allTokens.subList(0, maxLen - 1).toMutableList()
            result.add("[SEP]")  // Ensure [SEP] is always last
            result
        } else {
            allTokens
        }

        val tokenCount = truncatedTokens.size

        // Convert tokens → IDs
        val ids = LongArray(maxLen)
        val mask = LongArray(maxLen)

        for (i in truncatedTokens.indices) {
            ids[i] = (vocab[truncatedTokens[i]] ?: PhishGuardConfig.UNK_TOKEN_ID.toInt()).toLong()
            mask[i] = 1L
        }
        // Remaining positions are already 0 (PAD) in both arrays

        return TokenizeResult(
            inputIds = ids,
            attentionMask = mask,
            tokenCount = tokenCount,
            tokens = truncatedTokens
        )
    }

    /**
     * Convert token ID back to token string (for debugging parity tests).
     */
    fun idToToken(id: Int): String {
        return reverseVocab[id] ?: "[UNK]"
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BERT BasicTokenizer (Steps 1–5)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Full BERT BasicTokenizer pipeline.
     * Must match HuggingFace transformers BasicTokenizer(do_lower_case=True) exactly.
     */
    private fun basicTokenize(text: String): List<String> {
        // Step 1: Clean text — strip control chars, collapse whitespace
        var cleaned = cleanText(text)

        // Step 2: Insert spaces around CJK characters
        cleaned = tokenizeChineseChars(cleaned)

        // Step 3: Lowercase
        cleaned = cleaned.lowercase()

        // Step 4: Strip accents (NFD → remove combining marks)
        cleaned = stripAccents(cleaned)

        // Step 5: Split on punctuation (per Unicode category)
        val tokens = mutableListOf<String>()
        val whitespaceTokens = cleaned.trim().split(Regex("\\s+"))
        for (wsToken in whitespaceTokens) {
            if (wsToken.isNotEmpty()) {
                tokens.addAll(splitOnPunctuation(wsToken))
            }
        }

        return tokens.filter { it.isNotEmpty() }
    }

    /**
     * Step 1: Remove control characters and collapse whitespace.
     * Matches BERT's _clean_text() function.
     */
    private fun cleanText(text: String): String {
        val sb = StringBuilder(text.length)
        for (char in text) {
            val cp = char.code
            when {
                cp == 0 || cp == 0xFFFD || isControlChar(char) -> continue  // Skip
                isWhitespace(char) -> sb.append(' ')
                else -> sb.append(char)
            }
        }
        return sb.toString()
    }

    /**
     * Step 2: Add spaces around CJK Unified Ideographs.
     * Matches BERT's _tokenize_chinese_chars().
     * Kept enabled even for URLs to match HuggingFace behavior exactly.
     */
    private fun tokenizeChineseChars(text: String): String {
        val sb = StringBuilder(text.length + 16)
        for (char in text) {
            if (isChineseChar(char)) {
                sb.append(' ').append(char).append(' ')
            } else {
                sb.append(char)
            }
        }
        return sb.toString()
    }

    /**
     * Step 4: Strip accents by decomposing to NFD and removing combining marks.
     * Matches BERT's _run_strip_accents().
     */
    private fun stripAccents(text: String): String {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFD)
        val sb = StringBuilder(normalized.length)
        for (char in normalized) {
            if (Character.getType(char) != Character.NON_SPACING_MARK.toInt()) {
                sb.append(char)
            }
        }
        return sb.toString()
    }

    /**
     * Step 5: Split on punctuation characters.
     * Matches BERT's _run_split_on_punc().
     */
    private fun splitOnPunctuation(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val sb = StringBuilder()

        for (char in text) {
            if (isPunctuation(char)) {
                // Flush current buffer
                if (sb.isNotEmpty()) {
                    tokens.add(sb.toString())
                    sb.clear()
                }
                // Punctuation is its own token
                tokens.add(char.toString())
            } else {
                sb.append(char)
            }
        }

        if (sb.isNotEmpty()) {
            tokens.add(sb.toString())
        }

        return tokens
    }

    // ═══════════════════════════════════════════════════════════════════════
    // WordPiece Tokenizer (Step 6)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * WordPiece greedy longest-match tokenization.
     * Matches HuggingFace WordpieceTokenizer.tokenize() with ## prefix for sub-words.
     */
    private fun wordPieceTokenize(token: String): List<String> {
        if (token.isEmpty()) return emptyList()

        // Max word length — if longer, treat as [UNK]
        if (token.length > 200) {
            return listOf("[UNK]")
        }

        val subTokens = mutableListOf<String>()
        var start = 0
        var isBad = false

        while (start < token.length) {
            var end = token.length
            var curSubstr: String? = null

            while (start < end) {
                val substr = if (start > 0) {
                    "##" + token.substring(start, end)
                } else {
                    token.substring(start, end)
                }

                if (vocab.containsKey(substr)) {
                    curSubstr = substr
                    break
                }
                end--
            }

            if (curSubstr == null) {
                isBad = true
                break
            }

            subTokens.add(curSubstr)
            start = end
        }

        return if (isBad) listOf("[UNK]") else subTokens
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Character classification helpers (matching BERT's Python implementation)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Check if character is a whitespace character (BERT definition).
     */
    private fun isWhitespace(char: Char): Boolean {
        if (char == ' ' || char == '\t' || char == '\n' || char == '\r') return true
        val type = Character.getType(char)
        return type == Character.SPACE_SEPARATOR.toInt()
    }

    /**
     * Check if character is a control character (BERT definition).
     */
    private fun isControlChar(char: Char): Boolean {
        if (char == '\t' || char == '\n' || char == '\r') return false  // These are whitespace
        val type = Character.getType(char)
        return type == Character.CONTROL.toInt() || type == Character.FORMAT.toInt()
    }

    /**
     * Check if character is punctuation (BERT definition).
     * Uses Unicode categories + ASCII range check.
     */
    private fun isPunctuation(char: Char): Boolean {
        val cp = char.code
        // ASCII punctuation ranges
        if (cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126) {
            return true
        }
        val type = Character.getType(char)
        return type == Character.CONNECTOR_PUNCTUATION.toInt() ||
               type == Character.DASH_PUNCTUATION.toInt() ||
               type == Character.END_PUNCTUATION.toInt() ||
               type == Character.FINAL_QUOTE_PUNCTUATION.toInt() ||
               type == Character.INITIAL_QUOTE_PUNCTUATION.toInt() ||
               type == Character.OTHER_PUNCTUATION.toInt() ||
               type == Character.START_PUNCTUATION.toInt()
    }

    /**
     * Check if character is a CJK Unified Ideograph.
     * Matches BERT's _is_chinese_char().
     */
    private fun isChineseChar(char: Char): Boolean {
        val cp = char.code
        return (cp in 0x4E00..0x9FFF) ||
               (cp in 0x3400..0x4DBF) ||
               (cp in 0x20000..0x2A6DF) ||
               (cp in 0x2A700..0x2B73F) ||
               (cp in 0x2B740..0x2B81F) ||
               (cp in 0x2B820..0x2CEAF) ||
               (cp in 0xF900..0xFAFF) ||
               (cp in 0x2F800..0x2FA1F)
    }
}
