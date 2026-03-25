package com.cinedepth.pro.ui.retouch

import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Redo
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.cinedepth.pro.ui.BlurPreviewParams
import com.cinedepth.pro.ui.blur.DepthBlurEngine

private val AccentAmber = Color(0xFFFFC145)

@Composable
fun RetouchDepthScreen(
    state: RetouchDepthUiState,
    onBack: (() -> Unit)? = null,
    onImagePointClicked: (Int, Int) -> Unit = { _, _ -> },
    onPointerMoved: (Offset?) -> Unit = {},
    onStrokeStarted: (Offset, Color) -> Unit = { _, _ -> },
    onStrokeContinued: (Offset) -> Unit = {},
    onStrokeFinished: () -> Unit = {},
    onTransformChanged: (Float, Offset) -> Unit = { _, _ -> },
    onResetTransform: () -> Unit = {},
    onModeChange: (RetouchMode) -> Unit,
    onTargetDepthChange: (Int) -> Unit,
    onToggleDepthOverlay: (Boolean) -> Unit,
    onToggleTargetPicker: (Boolean) -> Unit,
    onManualBrushSizeChange: (Float) -> Unit,
    onFeatherRadiusChange: (Float) -> Unit,
    onAutoThresholdChange: (Float) -> Unit,
    onConfirmTargetDepth: () -> Unit = {},
    onGradientStartChange: (Int) -> Unit = {},
    onGradientEndChange: (Int) -> Unit = {},
    onUndo: () -> Unit,
    onRedo: () -> Unit = {},
    onClearSelection: () -> Unit = {},
    onFillSelection: () -> Unit = {},
    onApply: () -> Unit,
    onCancel: () -> Unit,
    onToggleTools: () -> Unit
) {
    val statusBarPadding = WindowInsets.statusBars.asPaddingValues()
    val navBarPadding = WindowInsets.navigationBars.asPaddingValues()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // Full-screen canvas
        DepthCanvasCard(
            state = state,
            onImagePointClicked = onImagePointClicked,
            onPointerMoved = onPointerMoved,
            onStrokeStarted = onStrokeStarted,
            onStrokeContinued = onStrokeContinued,
            onStrokeFinished = onStrokeFinished,
            onTransformChanged = onTransformChanged,
            modifier = Modifier.fillMaxSize()
        )

        // Top bar overlay
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color.Black.copy(alpha = 0.7f),
                            Color.Transparent
                        )
                    )
                )
                .padding(top = statusBarPadding.calculateTopPadding())
                .padding(horizontal = 4.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = { onBack?.invoke() }) {
                Icon(
                    Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = Color.White
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            // Status pill
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(Color(0xCC1A1A1A))
                    .padding(horizontal = 14.dp, vertical = 7.dp)
            ) {
                val zoomText = if (state.zoomScale > 1.01f)
                    "${(state.zoomScale * 100).toInt()}%  |  " else ""
                Text(
                    zoomText + state.canvasStatus,
                    color = Color.White.copy(alpha = 0.8f),
                    style = MaterialTheme.typography.labelSmall
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            if (state.zoomScale > 1.01f || state.panOffset != Offset.Zero) {
                TextButton(onClick = onResetTransform) {
                    Text(
                        "Reset",
                        color = AccentAmber,
                        style = MaterialTheme.typography.labelMedium
                    )
                }
            }

            // Undo
            IconButton(onClick = onUndo, enabled = state.undoAvailable) {
                Icon(
                    Icons.AutoMirrored.Filled.Undo,
                    contentDescription = "Undo",
                    tint = if (state.undoAvailable) Color.White else Color.White.copy(alpha = 0.25f),
                    modifier = Modifier.size(22.dp)
                )
            }

            // Redo
            IconButton(onClick = onRedo, enabled = state.redoAvailable) {
                Icon(
                    Icons.AutoMirrored.Filled.Redo,
                    contentDescription = "Redo",
                    tint = if (state.redoAvailable) Color.White else Color.White.copy(alpha = 0.25f),
                    modifier = Modifier.size(22.dp)
                )
            }

            // Apply
            IconButton(onClick = onApply) {
                Icon(
                    Icons.Default.Check,
                    contentDescription = "Apply",
                    tint = AccentAmber
                )
            }
        }

        // FAB for tools
        AnimatedVisibility(
            visible = !state.showTools,
            enter = fadeIn(),
            exit = fadeOut(),
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(
                    bottom = navBarPadding.calculateBottomPadding() + 24.dp,
                    end = 20.dp
                )
        ) {
            Box(
                modifier = Modifier
                    .size(52.dp)
                    .clip(CircleShape)
                    .background(AccentAmber)
                    .clickable(onClick = onToggleTools),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Outlined.Tune,
                    contentDescription = "Tools",
                    tint = Color(0xFF1A1200),
                    modifier = Modifier.size(24.dp)
                )
            }
        }

        // Bottom tools panel
        AnimatedVisibility(
            visible = state.showTools,
            enter = fadeIn() + slideInVertically { it },
            exit = fadeOut() + slideOutVertically { it },
            modifier = Modifier.align(Alignment.BottomCenter)
        ) {
            RetouchControls(
                state = state,
                navBarPadding = navBarPadding.calculateBottomPadding(),
                onModeChange = onModeChange,
                onTargetDepthChange = onTargetDepthChange,
                onToggleDepthOverlay = onToggleDepthOverlay,
                onToggleTargetPicker = onToggleTargetPicker,
                onManualBrushSizeChange = onManualBrushSizeChange,
                onFeatherRadiusChange = onFeatherRadiusChange,
                onAutoThresholdChange = onAutoThresholdChange,
                onConfirmTargetDepth = onConfirmTargetDepth,
                onGradientStartChange = onGradientStartChange,

                onGradientEndChange = onGradientEndChange,
                onClearSelection = onClearSelection,
                onFillSelection = onFillSelection,
                onCancel = onCancel,
                onApply = onApply,
                onToggleTools = onToggleTools
            )
        }
    }
}

@Composable
private fun DepthCanvasCard(
    state: RetouchDepthUiState,
    onImagePointClicked: (Int, Int) -> Unit = { _, _ -> },
    onPointerMoved: (Offset?) -> Unit = {},
    onStrokeStarted: (Offset, Color) -> Unit = { _, _ -> },
    onStrokeContinued: (Offset) -> Unit = {},
    onStrokeFinished: () -> Unit = {},
    onTransformChanged: (Float, Offset) -> Unit = { _, _ -> },
    modifier: Modifier = Modifier
) {
    val density = LocalDensity.current
    val touchOffsetY = with(density) { -50.dp.toPx() }

    Box(modifier = modifier
        .background(Color(0xFF080808))
        .pointerInput(Unit) {
            detectTransformGestures { _, pan, zoom, _ ->
                onTransformChanged(zoom, pan)
            }
        }
    ) {
        val strokeColor = depthColorForValue(state.targetDepth)

        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(state.mode, state.targetDepth, state.zoomScale, state.panOffset) {
                    fun mapToCanvasContent(position: Offset): Offset {
                        val canvasCenter = Offset(size.width / 2f, size.height / 2f)
                        return ((position - state.panOffset - canvasCenter) / state.zoomScale) + canvasCenter
                    }

                    fun mapToImagePoint(position: Offset): Pair<Int, Int>? {
                        val contentPos = mapToCanvasContent(position)
                        val bitmap = state.sourceBitmap ?: state.depthMapBitmap ?: return null
                        val imgW = bitmap.width.toFloat()
                        val imgH = bitmap.height.toFloat()
                        val scale = minOf(size.width / imgW, size.height / imgH)
                        val drawW = imgW * scale
                        val drawH = imgH * scale
                        val offsetX = (size.width - drawW) / 2f
                        val offsetY = (size.height - drawH) / 2f
                        if (
                            contentPos.x < offsetX ||
                            contentPos.x > offsetX + drawW ||
                            contentPos.y < offsetY ||
                            contentPos.y > offsetY + drawH
                        ) return null

                        val imageX = ((contentPos.x - offsetX) / drawW * imgW)
                            .toInt()
                            .coerceIn(0, bitmap.width - 1)
                        val imageY = ((contentPos.y - offsetY) / drawH * imgH)
                            .toInt()
                            .coerceIn(0, bitmap.height - 1)
                        return imageX to imageY
                    }

                    if (
                        state.mode == RetouchMode.ManualPaint ||
                        state.mode == RetouchMode.Erase ||
                        state.mode == RetouchMode.GradientBrush
                    ) {
                        detectDragGestures(
                            onDragStart = { offset ->
                                val rawPaintPos = offset.copy(y = offset.y + touchOffsetY)
                                val mappedPos = mapToCanvasContent(rawPaintPos)
                                onPointerMoved(rawPaintPos)
                                onStrokeStarted(mappedPos, strokeColor)
                            },
                            onDrag = { change, _ ->
                                val rawPaintPos = change.position.copy(y = change.position.y + touchOffsetY)
                                val mappedPos = mapToCanvasContent(rawPaintPos)
                                onPointerMoved(rawPaintPos)
                                onStrokeContinued(mappedPos)
                            },
                            onDragEnd = {
                                onPointerMoved(null)
                                onStrokeFinished()
                            },
                            onDragCancel = {
                                onPointerMoved(null)
                                onStrokeFinished()
                            }
                        )
                    } else {
                        detectTapGestures { offset ->
                            mapToImagePoint(offset)?.let { (imageX, imageY) ->
                                onImagePointClicked(imageX, imageY)
                            }
                        }
                    }
                }
        ) {
            val canvasW = size.width
            val canvasH = size.height
            val canvasCenter = Offset(canvasW / 2f, canvasH / 2f)

            withTransform({
                scale(state.zoomScale, state.zoomScale, pivot = canvasCenter)
                translate(state.panOffset.x, state.panOffset.y)
            }) {
                val img = state.sourceBitmap
                if (img != null) {
                    val imgW = img.width.toFloat()
                    val imgH = img.height.toFloat()
                    val scale = minOf(canvasW / imgW, canvasH / imgH)
                    val drawW = imgW * scale
                    val drawH = imgH * scale
                    val offsetX = (canvasW - drawW) / 2f
                    val offsetY = (canvasH - drawH) / 2f
                    val drawRect = Rect(offsetX, offsetY, offsetX + drawW, offsetY + drawH)

                    clipRect(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom) {
                        state.depthMapBitmap?.let {
                            drawImage(
                                it.asImageBitmap(),
                                dstSize = IntSize(drawW.toInt(), drawH.toInt()),
                                dstOffset = androidx.compose.ui.unit.IntOffset(offsetX.toInt(), offsetY.toInt())
                            )
                        }
                        if (!state.showDepthOverlay) {
                            drawImage(
                                img.asImageBitmap(),
                                dstSize = IntSize(drawW.toInt(), drawH.toInt()),
                                dstOffset = androidx.compose.ui.unit.IntOffset(offsetX.toInt(), offsetY.toInt()),
                                alpha = 0.4f
                            )
                        }
                        state.selectionMask?.let {
                            drawImage(
                                it.asImageBitmap(),
                                dstSize = IntSize(drawW.toInt(), drawH.toInt()),
                                dstOffset = androidx.compose.ui.unit.IntOffset(offsetX.toInt(), offsetY.toInt())
                            )
                        }
                        state.smartSelectPreviewMask?.let {
                            drawImage(
                                it.asImageBitmap(),
                                dstSize = IntSize(drawW.toInt(), drawH.toInt()),
                                dstOffset = androidx.compose.ui.unit.IntOffset(offsetX.toInt(), offsetY.toInt())
                            )
                        }
                        for (stroke in state.strokes) {
                            if (stroke.points.size > 1) {
                                val path = Path().apply {
                                    moveTo(stroke.points.first().x, stroke.points.first().y)
                                    for (i in 1 until stroke.points.size) {
                                        lineTo(stroke.points[i].x, stroke.points[i].y)
                                    }
                                }
                                drawPath(
                                    path = path,
                                    color = stroke.color,
                                    style = Stroke(
                                        width = stroke.size,
                                        cap = StrokeCap.Round,
                                        join = androidx.compose.ui.graphics.StrokeJoin.Round
                                    )
                                )
                            } else if (stroke.points.size == 1) {
                                drawCircle(
                                    color = stroke.color,
                                    radius = stroke.size / 2f,
                                    center = stroke.points.first()
                                )
                            }
                        }
                    }
                }
            }

            // Brush cursor
            if (
                state.mode == RetouchMode.ManualPaint ||
                state.mode == RetouchMode.Erase ||
                state.mode == RetouchMode.GradientBrush
            ) {
                state.lastTouchPoint?.let { pos ->
                    val brushRadius = (state.manualBrushSize / 2f) * state.zoomScale
                    val gray = state.targetDepth / 255f
                    val fillColor = if (state.mode == RetouchMode.Erase) {
                        Color(0xFFFFF2D2).copy(alpha = 0.26f)
                    } else {
                        Color(gray, gray, gray, 0.35f)
                    }
                    val ringColor = if (state.mode == RetouchMode.Erase) {
                        Color(0xFFFF8A4C)
                    } else {
                        AccentAmber
                    }
                    drawCircle(
                        color = fillColor,
                        radius = brushRadius,
                        center = pos
                    )
                    drawCircle(
                        color = ringColor.copy(alpha = 0.82f),
                        radius = brushRadius,
                        center = pos,
                        style = Stroke(width = 1.5.dp.toPx())
                    )
                    drawCircle(
                        color = ringColor,
                        radius = 2.dp.toPx(),
                        center = pos
                    )
                }
            }
        }
    }
}

@Composable
private fun RetouchControls(
    state: RetouchDepthUiState,
    navBarPadding: androidx.compose.ui.unit.Dp,
    onModeChange: (RetouchMode) -> Unit,
    onTargetDepthChange: (Int) -> Unit,
    onToggleDepthOverlay: (Boolean) -> Unit,
    onToggleTargetPicker: (Boolean) -> Unit,
    onManualBrushSizeChange: (Float) -> Unit,
    onFeatherRadiusChange: (Float) -> Unit,
    onAutoThresholdChange: (Float) -> Unit,
    onConfirmTargetDepth: () -> Unit = {},
    onGradientStartChange: (Int) -> Unit = {},
    onGradientEndChange: (Int) -> Unit = {},
    onClearSelection: () -> Unit = {},
    onFillSelection: () -> Unit = {},
    onCancel: () -> Unit,
    onApply: () -> Unit,
    onToggleTools: () -> Unit
) {
    val selectGroupActive =
        state.mode == RetouchMode.SmartSelect ||
            state.mode == RetouchMode.AutoSegment
    val drawGroupActive =
        state.mode == RetouchMode.ManualPaint ||
            state.mode == RetouchMode.Erase ||
            state.mode == RetouchMode.GradientBrush

    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
        color = Color(0xE6141414),
        tonalElevation = 0.dp
    ) {
        Column(
            modifier = Modifier
                .padding(top = 8.dp, start = 16.dp, end = 16.dp)
                .padding(bottom = navBarPadding + 6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Box(
                modifier = Modifier
                    .width(32.dp)
                    .height(3.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(Color.White.copy(alpha = 0.2f))
                    .align(Alignment.CenterHorizontally)
                    .clickable(onClick = onToggleTools)
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                RetouchModeChip(
                    label = "Select",
                    selected = selectGroupActive,
                    onClick = {
                        if (!selectGroupActive) {
                            onModeChange(RetouchMode.SmartSelect)
                        }
                    },
                    modifier = Modifier.weight(1f)
                )
                RetouchModeChip(
                    label = "Draw",
                    selected = drawGroupActive,
                    onClick = {
                        if (!drawGroupActive) {
                            onModeChange(RetouchMode.ManualPaint)
                        }
                    },
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (drawGroupActive) {
                    RetouchSubModeChip(
                        label = "Paint",
                        selected = state.mode == RetouchMode.ManualPaint,
                        onClick = { onModeChange(RetouchMode.ManualPaint) },
                        modifier = Modifier.weight(1f)
                    )
                    RetouchSubModeChip(
                        label = "Erase",
                        selected = state.mode == RetouchMode.Erase,
                        onClick = { onModeChange(RetouchMode.Erase) },
                        modifier = Modifier.weight(1f)
                    )
                    RetouchSubModeChip(
                        label = "Gradient",
                        selected = state.mode == RetouchMode.GradientBrush,
                        onClick = { onModeChange(RetouchMode.GradientBrush) },
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    "Depth ${state.targetDepth}",
                    color = Color.White,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Medium
                )
                Spacer(Modifier.width(8.dp))
                val gray = state.targetDepth / 255f
                Box(
                    modifier = Modifier
                        .size(18.dp)
                        .clip(CircleShape)
                        .background(Color(gray, gray, gray))
                        .border(1.dp, AccentAmber.copy(alpha = 0.5f), CircleShape)
                )
                Spacer(Modifier.weight(1f))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        if (state.targetPickerEnabled) "Picking…" else "Pick",
                        color = if (state.targetPickerEnabled) AccentAmber else Color.White.copy(alpha = 0.6f),
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(
                                if (state.targetPickerEnabled) AccentAmber.copy(alpha = 0.18f)
                                else Color.White.copy(alpha = 0.08f)
                            )
                            .clickable { onToggleTargetPicker(!state.targetPickerEnabled) }
                            .padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                    Text(
                        "Photo",
                        color = Color.White.copy(alpha = 0.6f),
                        style = MaterialTheme.typography.labelSmall
                    )
                    Switch(
                        checked = !state.showDepthOverlay,
                        onCheckedChange = { onToggleDepthOverlay(!it) },
                        modifier = Modifier.height(22.dp),
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = AccentAmber,
                            checkedTrackColor = AccentAmber.copy(alpha = 0.3f)
                        )
                    )
                }
            }

            // Mode-specific controls
            when (state.mode) {
                RetouchMode.ManualPaint -> {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        CompactRetouchSlider(
                            label = "Brush",
                            valueLabel = state.manualBrushSize.toInt().toString(),
                            value = state.manualBrushSize,
                            onValueChange = onManualBrushSizeChange,
                            valueRange = 10f..250f,
                            modifier = Modifier.weight(1f)
                        )
                        CompactRetouchSlider(
                            label = "Soft",
                            valueLabel = state.featherRadiusPx.toInt().toString(),
                            value = state.featherRadiusPx,
                            onValueChange = onFeatherRadiusChange,
                            valueRange = 2f..36f,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    RetouchSlider(
                        label = "Draw Depth ${state.targetDepth}",
                        value = state.targetDepth.toFloat(),
                        onValueChange = { onTargetDepthChange(it.toInt()) },
                        valueRange = 0f..255f
                    )
                }

                RetouchMode.Erase -> {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        CompactRetouchSlider(
                            label = "Brush",
                            valueLabel = state.manualBrushSize.toInt().toString(),
                            value = state.manualBrushSize,
                            onValueChange = onManualBrushSizeChange,
                            valueRange = 10f..250f,
                            modifier = Modifier.weight(1f)
                        )
                        CompactRetouchSlider(
                            label = "Soft",
                            valueLabel = state.featherRadiusPx.toInt().toString(),
                            value = state.featherRadiusPx,
                            onValueChange = onFeatherRadiusChange,
                            valueRange = 2f..36f,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Text(
                        text = "Erase restores the original depth and feathers the contour.",
                        color = Color.White.copy(alpha = 0.72f),
                        style = MaterialTheme.typography.labelSmall
                    )
                }

                RetouchMode.GradientBrush -> {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        CompactRetouchSlider(
                            label = "Brush",
                            valueLabel = state.manualBrushSize.toInt().toString(),
                            value = state.manualBrushSize,
                            onValueChange = onManualBrushSizeChange,
                            valueRange = 20f..300f,
                            modifier = Modifier.weight(1f)
                        )
                        CompactRetouchSlider(
                            label = "Near",
                            valueLabel = state.gradientStartDepth.toString(),
                            value = state.gradientStartDepth.toFloat(),
                            onValueChange = { onGradientStartChange(it.toInt()) },
                            valueRange = 0f..255f,
                            modifier = Modifier.weight(1f)
                        )
                        CompactRetouchSlider(
                            label = "Far",
                            valueLabel = state.gradientEndDepth.toString(),
                            value = state.gradientEndDepth.toFloat(),
                            onValueChange = { onGradientEndChange(it.toInt()) },
                            valueRange = 0f..255f,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }

                RetouchMode.SmartSelect, RetouchMode.AutoSegment -> {
                    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                        Text(
                            text = if (state.targetPickerEnabled) {
                                "Tap image to sample target depth."
                            } else {
                                "Tap area, then confirm and fill. Use Pick to sample depth."
                            },
                            color = Color.White.copy(alpha = 0.68f),
                            style = MaterialTheme.typography.labelSmall
                        )
                        CompactRetouchSlider(
                            label = "Sense",
                            valueLabel = "${(state.autoSegmentThreshold * 100f).toInt()}%",
                            value = state.autoSegmentThreshold,
                            onValueChange = onAutoThresholdChange,
                            valueRange = 0.01f..0.15f
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            Button(
                                onClick = onConfirmTargetDepth,
                                enabled = state.smartSelectPreviewMask != null,
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (state.targetDepthConfirmed) Color.Gray.copy(alpha = 0.3f) else Color(0xFF5CD6B3),
                                    contentColor = Color.Black
                                ),
                                modifier = Modifier
                                    .weight(1f)
                                    .height(34.dp)
                            ) {
                                Text(if (state.targetDepthConfirmed) "Added" else "Confirm")
                            }
                            Button(
                                onClick = onClearSelection,
                                enabled = state.selectionMask != null || state.smartSelectPreviewMask != null,
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color.White.copy(alpha = 0.12f)
                                ),
                                modifier = Modifier
                                    .weight(1f)
                                    .height(34.dp)
                            ) {
                                Text("Clear")
                            }
                            Button(
                                onClick = onFillSelection,
                                enabled = state.selectionMask != null,
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = AccentAmber,
                                    contentColor = Color.Black
                                ),
                                modifier = Modifier
                                    .weight(1f)
                                    .height(34.dp)
                            ) {
                                Text("Fill")
                            }
                        }
                    }
                }

                RetouchMode.Eyedropper -> Unit
            }

            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                TextButton(
                    onClick = onCancel,
                    modifier = Modifier
                        .weight(1f)
                        .height(38.dp)
                ) {
                    Text(
                        "Discard",
                        color = Color.White.copy(alpha = 0.5f),
                        style = MaterialTheme.typography.labelMedium
                    )
                }
                Button(
                    onClick = onApply,
                    modifier = Modifier
                        .weight(1f)
                        .height(38.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = AccentAmber,
                        contentColor = Color(0xFF1A1200)
                    )
                ) {
                    Text(
                        "Apply",
                        fontWeight = FontWeight.Bold,
                        style = MaterialTheme.typography.labelMedium
                    )
                }
            }
        }
    }
}

@Composable
private fun RetouchSubModeChip(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(
                if (selected) AccentAmber.copy(alpha = 0.12f)
                else Color.White.copy(alpha = 0.04f)
            )
            .then(
                if (selected) Modifier.border(1.dp, AccentAmber.copy(alpha = 0.32f), RoundedCornerShape(10.dp))
                else Modifier
            )
            .clickable(onClick = onClick)
            .padding(vertical = 5.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = label,
            color = if (selected) AccentAmber else Color.White.copy(alpha = 0.7f),
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
private fun RetouchModeChip(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(
                if (selected) AccentAmber.copy(alpha = 0.15f)
                else Color.White.copy(alpha = 0.06f)
            )
            .then(
                if (selected) Modifier.border(1.dp, AccentAmber.copy(alpha = 0.4f), RoundedCornerShape(12.dp))
                else Modifier
            )
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelMedium,
            color = if (selected) AccentAmber else Color.White.copy(alpha = 0.5f),
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal
        )
    }
}

@Composable
private fun CompactRetouchSlider(
    label: String,
    valueLabel: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    valueRange: ClosedFloatingPointRange<Float>,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                color = Color.White.copy(alpha = 0.6f),
                style = MaterialTheme.typography.labelSmall
            )
            Text(
                text = valueLabel,
                color = AccentAmber,
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Medium
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            modifier = Modifier.height(26.dp),
            colors = SliderDefaults.colors(
                thumbColor = AccentAmber,
                activeTrackColor = AccentAmber,
                inactiveTrackColor = Color.White.copy(alpha = 0.08f)
            )
        )
    }
}

@Composable
private fun RetouchSlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    valueRange: ClosedFloatingPointRange<Float>
) {
    Column {
        Text(
            label,
            color = Color.White.copy(alpha = 0.5f),
            style = MaterialTheme.typography.labelSmall
        )
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            modifier = Modifier.height(32.dp),
            colors = SliderDefaults.colors(
                thumbColor = AccentAmber,
                activeTrackColor = AccentAmber,
                inactiveTrackColor = Color.White.copy(alpha = 0.08f)
            )
        )
    }
}

private fun depthColorForValue(value: Int): Color {
    val t = (value.coerceIn(0, 255) / 255f)
    return Color(t, t, t, 1f)
}

@Composable
fun RetouchDepthRoute(
    selectedPhotoUriString: String? = null,
    blurPreviewParams: BlurPreviewParams? = null,
    highQualityDepthEnabled: Boolean = false,
    existingDepth: Bitmap? = null,
    forceFreshDepth: Boolean = false,
    onApplyRetouch: () -> Unit = {},
    onBack: (() -> Unit)? = null,
    viewModel: RetouchDepthViewModel = viewModel()
) {
    val state = viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current

    LaunchedEffect(selectedPhotoUriString, forceFreshDepth, highQualityDepthEnabled) {
        selectedPhotoUriString?.let {
            viewModel.onAction(
                RetouchDepthAction.LoadInitialData(
                    context = context,
                    uri = Uri.parse(it),
                    existingDepth = existingDepth,
                    forceFreshDepth = forceFreshDepth,
                    highQualityDepth = highQualityDepthEnabled
                )
            )
        }
    }

    LaunchedEffect(blurPreviewParams) {
        if (blurPreviewParams != null) {
            viewModel.onAction(RetouchDepthAction.BlurPreviewDefaultsLoaded(blurPreviewParams))
        }
    }

    RetouchDepthScreen(
        state = state.value,
        onBack = {
            viewModel.onAction(RetouchDepthAction.ExitRequested)
            onBack?.invoke()
        },
        onImagePointClicked = { imageX, imageY -> viewModel.onAction(RetouchDepthAction.ImagePointClicked(imageX, imageY)) },
        onPointerMoved = { viewModel.onAction(RetouchDepthAction.PointerMoved(it)) },
        onStrokeStarted = { offset, color -> viewModel.onAction(RetouchDepthAction.StrokeStarted(offset, color)) },
        onStrokeContinued = { offset -> viewModel.onAction(RetouchDepthAction.StrokeContinued(offset)) },
        onStrokeFinished = { viewModel.onAction(RetouchDepthAction.StrokeFinished) },
        onTransformChanged = { zoom, pan -> viewModel.onAction(RetouchDepthAction.TransformChanged(zoom, pan)) },
        onResetTransform = { viewModel.onAction(RetouchDepthAction.ResetTransformClicked) },
        onModeChange = { viewModel.onAction(RetouchDepthAction.ModeChanged(it)) },
        onTargetDepthChange = { viewModel.onAction(RetouchDepthAction.TargetDepthChanged(it)) },
        onToggleDepthOverlay = { viewModel.onAction(RetouchDepthAction.DepthOverlayToggled(it)) },
        onToggleTargetPicker = { viewModel.onAction(RetouchDepthAction.TargetPickerToggled(it)) },
        onManualBrushSizeChange = { viewModel.onAction(RetouchDepthAction.ManualBrushSizeChanged(it)) },
        onFeatherRadiusChange = { viewModel.onAction(RetouchDepthAction.FeatherRadiusChanged(it)) },
        onAutoThresholdChange = { viewModel.onAction(RetouchDepthAction.AutoThresholdChanged(it)) },
        onConfirmTargetDepth = { viewModel.onAction(RetouchDepthAction.ConfirmTargetDepthClicked) },
        onGradientStartChange = { viewModel.onAction(RetouchDepthAction.GradientStartChanged(it)) },
        onGradientEndChange = { viewModel.onAction(RetouchDepthAction.GradientEndChanged(it)) },
        onUndo = { viewModel.onAction(RetouchDepthAction.UndoClicked) },
        onRedo = { viewModel.onAction(RetouchDepthAction.RedoClicked) },
        onClearSelection = { viewModel.onAction(RetouchDepthAction.ClearSelectionClicked) },
        onFillSelection = { viewModel.onAction(RetouchDepthAction.FillSelectionClicked) },
        onApply = {
            viewModel.onAction(RetouchDepthAction.ApplyClicked { baked ->
                DepthBlurEngine.setModifiedDepth(baked)
                onApplyRetouch()
            })
        },
        onCancel = { viewModel.onAction(RetouchDepthAction.CancelClicked) },
        onToggleTools = { viewModel.onAction(RetouchDepthAction.ToggleToolsClicked) }
    )
}
