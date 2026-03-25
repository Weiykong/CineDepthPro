package com.cinedepth.pro.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

// Accent: warm amber / gold for a photography-app feel
private val Amber = Color(0xFFFFC145)
private val AmberDark = Color(0xFFE5A832)
private val AmberContainer = Color(0xFF3D2E00)

private val LightColors = lightColorScheme(
    primary = Color(0xFF8B6800),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFFFDF9E),
    onPrimaryContainer = Color(0xFF2B1F00),
    secondary = Color(0xFF6B5E3F),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFF5E1B9),
    onSecondaryContainer = Color(0xFF241B04),
    tertiary = Color(0xFF4C6545),
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFCEEBC3),
    onTertiaryContainer = Color(0xFF092007),
    background = Color(0xFFFFFBFF),
    surface = Color(0xFFFFFBFF),
    surfaceVariant = Color(0xFFEDE1CF),
    onSurface = Color(0xFF1E1B16),
    onSurfaceVariant = Color(0xFF4D4639),
    outline = Color(0xFF7F7668),
    outlineVariant = Color(0xFFD0C5B4)
)

private val DarkColors = darkColorScheme(
    primary = Amber,
    onPrimary = Color(0xFF1E1200),
    primaryContainer = AmberContainer,
    onPrimaryContainer = Color(0xFFFFDF9E),
    secondary = Color(0xFFD8C59F),
    onSecondary = Color(0xFF3B2F15),
    secondaryContainer = Color(0xFF52462A),
    onSecondaryContainer = Color(0xFFF5E1B9),
    tertiary = Color(0xFFB2CFA8),
    onTertiary = Color(0xFF1F361B),
    tertiaryContainer = Color(0xFF354D30),
    onTertiaryContainer = Color(0xFFCEEBC3),
    background = Color(0xFF111111),
    surface = Color(0xFF141414),
    surfaceVariant = Color(0xFF2A2A2A),
    onSurface = Color(0xFFEAE1D9),
    onSurfaceVariant = Color(0xFFD0C5B4),
    outline = Color(0xFF998F80),
    outlineVariant = Color(0xFF4D4639)
)

private val AppTypography = Typography(
    headlineLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Bold,
        fontSize = 30.sp,
        lineHeight = 38.sp,
        letterSpacing = (-0.5).sp
    ),
    headlineMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 24.sp,
        lineHeight = 32.sp
    ),
    titleLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 20.sp,
        lineHeight = 28.sp
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.15.sp
    ),
    bodyLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.5.sp
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.25.sp
    ),
    labelLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.1.sp
    ),
    labelMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 12.sp,
        lineHeight = 16.sp,
        letterSpacing = 0.5.sp
    )
)

@Composable
fun CineDepthTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        content = content
    )
}
