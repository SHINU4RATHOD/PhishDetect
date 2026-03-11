package com.phishguard.app.ui.theme

import androidx.compose.ui.graphics.Color

// ── Primary Palette ──────────────────────────────────────────────────────────
val PhishGuardBlue = Color(0xFF1A73E8)           // Primary brand
val PhishGuardBlueDark = Color(0xFF1557B0)       // Pressed state
val PhishGuardBlueLight = Color(0xFF4A90D9)      // Hover state

// ── Security Verdict ──────────────────────────────────────────────────────────
val SafeGreen = Color(0xFF00C853)                // SAFE verdict
val SafeGreenSurface = Color(0x1A00C853)         // Green surface (10% opacity)
val SafeGreenDark = Color(0xFF2E7D32)            // Dark mode SAFE

val DangerRed = Color(0xFFFF1744)                // PHISHING verdict
val DangerRedSurface = Color(0x1AFF1744)         // Red surface (10% opacity)
val DangerRedDark = Color(0xFFD32F2F)            // Dark mode PHISHING

val WarningAmber = Color(0xFFFFAB00)             // Warning/caution
val WarningAmberSurface = Color(0x1AFFAB00)      // Amber surface

// ── Dark Theme ───────────────────────────────────────────────────────────────
val DarkBackground = Color(0xFF0D1117)           // GitHub-dark inspired
val DarkSurface = Color(0xFF161B22)              // Card surface
val DarkSurfaceElevated = Color(0xFF21262D)      // Elevated surface
val DarkOnSurface = Color(0xFFC9D1D9)            // Text on dark
val DarkOnSurfaceSecondary = Color(0xFF8B949E)   // Secondary text
val DarkBorder = Color(0xFF30363D)               // Borders

// ── Light Theme ──────────────────────────────────────────────────────────────
val LightBackground = Color(0xFFF6F8FA)          // Soft light bg
val LightSurface = Color(0xFFFFFFFF)             // Card surface
val LightSurfaceElevated = Color(0xFFF0F2F5)     // Elevated
val LightOnSurface = Color(0xFF1F2937)           // Text on light
val LightOnSurfaceSecondary = Color(0xFF6B7280)  // Secondary text
val LightBorder = Color(0xFFE5E7EB)              // Borders

// ── Accent ───────────────────────────────────────────────────────────────────
val AccentCyan = Color(0xFF00BCD4)               // Accent highlights
val AccentPurple = Color(0xFF7C3AED)             // Premium accent
