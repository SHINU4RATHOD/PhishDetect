package com.phishguard.app.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val DarkColorScheme = darkColorScheme(
    primary = PhishGuardBlue,
    onPrimary = Color.White,
    primaryContainer = PhishGuardBlueDark,
    onPrimaryContainer = Color.White,
    secondary = AccentCyan,
    onSecondary = Color.Black,
    tertiary = AccentPurple,
    onTertiary = Color.White,
    background = DarkBackground,
    onBackground = DarkOnSurface,
    surface = DarkSurface,
    onSurface = DarkOnSurface,
    surfaceVariant = DarkSurfaceElevated,
    onSurfaceVariant = DarkOnSurfaceSecondary,
    outline = DarkBorder,
    error = DangerRed,
    onError = Color.White
)

private val LightColorScheme = lightColorScheme(
    primary = PhishGuardBlue,
    onPrimary = Color.White,
    primaryContainer = PhishGuardBlueLight,
    onPrimaryContainer = Color.White,
    secondary = AccentCyan,
    onSecondary = Color.Black,
    tertiary = AccentPurple,
    onTertiary = Color.White,
    background = LightBackground,
    onBackground = LightOnSurface,
    surface = LightSurface,
    onSurface = LightOnSurface,
    surfaceVariant = LightSurfaceElevated,
    onSurfaceVariant = LightOnSurfaceSecondary,
    outline = LightBorder,
    error = DangerRed,
    onError = Color.White
)

@Composable
fun PhishGuardTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            if (darkTheme) dynamicDarkColorScheme(LocalView.current.context)
            else dynamicLightColorScheme(LocalView.current.context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    // Set status bar color to match theme
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = PhishGuardTypography,
        content = content
    )
}
