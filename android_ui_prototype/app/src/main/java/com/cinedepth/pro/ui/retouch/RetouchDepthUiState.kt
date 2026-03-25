package com.cinedepth.pro.ui.retouch

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import android.graphics.Bitmap

data class Stroke(
    val points: List<Offset>,
    val size: Float,
    val color: Color,
    val opacity: Float,
    val isErase: Boolean = false,
    val isGradient: Boolean = false,
    val gradientStartDepth: Int = 0,
    val gradientEndDepth: Int = 255
)

enum class RetouchMode {
    ManualPaint,
    Erase,
    GradientBrush,
    SmartSelect,
    AutoSegment,
    Eyedropper
}

data class RetouchDepthUiState(
    val mode: RetouchMode = RetouchMode.SmartSelect,
    val targetDepth: Int = 128,
    val targetDepthConfirmed: Boolean = false,
    val showDepthOverlay: Boolean = true,
    val targetPickerEnabled: Boolean = false,
    val edgeSnap: Boolean = true,
    val featherRadiusPx: Float = 10f,
    val undoAvailable: Boolean = false,
    val redoAvailable: Boolean = false,
    val manualBrushSize: Float = 56f,
    val manualBrushHardness: Float = 0.7f,
    val manualBrushOpacity: Float = 1f,
    val gradientStartDepth: Int = 0,
    val gradientEndDepth: Int = 255,
    val autoSegmentPrompt: String = "Tap area to select. Use slider to set depth and tap Fill.",
    val autoSegmentSelectedLabel: String? = null,
    val autoSegmentThreshold: Float = 0.04f,
    val autoSegmentBusy: Boolean = false,
    val autoSegmentMaskPreview: Boolean = true,
    val canvasStatus: String = "Loading depth map...",
    val strokes: List<Stroke> = emptyList(),
    val undoneStrokes: List<Stroke> = emptyList(),
    val sourceBitmap: Bitmap? = null,
    val depthMapBitmap: Bitmap? = null,
    val rawDepthData: FloatArray? = null,
    val selectionMask: Bitmap? = null,
    val smartSelectPreviewMask: Bitmap? = null,
    val lastSmartSelectPoint: Offset? = null,
    val showTools: Boolean = true,
    val lastTouchPoint: Offset? = null,
    val zoomScale: Float = 1f,
    val panOffset: Offset = Offset.Zero
)
