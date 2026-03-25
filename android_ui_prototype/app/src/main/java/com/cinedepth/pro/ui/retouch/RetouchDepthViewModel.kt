package com.cinedepth.pro.ui.retouch

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.net.Uri
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.cinedepth.pro.ui.BlurPreviewParams
import com.cinedepth.pro.ui.blur.DepthBlurEngine
import com.cinedepth.pro.ui.blur.DepthEstimator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface RetouchDepthAction {
    data class LoadInitialData(
        val context: Context,
        val uri: Uri,
        val existingDepth: Bitmap? = null,
        val forceFreshDepth: Boolean = false,
        val highQualityDepth: Boolean = false
    ) : RetouchDepthAction
    data class BlurPreviewDefaultsLoaded(val params: BlurPreviewParams) : RetouchDepthAction
    data class ModeChanged(val mode: RetouchMode) : RetouchDepthAction
    data class TargetDepthChanged(val value: Int) : RetouchDepthAction
    data class DepthOverlayToggled(val enabled: Boolean) : RetouchDepthAction
    data class TargetPickerToggled(val enabled: Boolean) : RetouchDepthAction
    data class EdgeSnapToggled(val enabled: Boolean) : RetouchDepthAction
    data class FeatherRadiusChanged(val value: Float) : RetouchDepthAction
    data class ManualBrushSizeChanged(val value: Float) : RetouchDepthAction
    data class ManualBrushHardnessChanged(val value: Float) : RetouchDepthAction
    data class ManualBrushOpacityChanged(val value: Float) : RetouchDepthAction
    data class GradientStartChanged(val value: Int) : RetouchDepthAction
    data class GradientEndChanged(val value: Int) : RetouchDepthAction
    data class AutoThresholdChanged(val value: Float) : RetouchDepthAction
    data class AutoMaskPreviewToggled(val enabled: Boolean) : RetouchDepthAction
    data object AutoSegmentDetectRequested : RetouchDepthAction
    
    data class ImagePointClicked(val imageX: Int, val imageY: Int) : RetouchDepthAction
    data class StrokeStarted(val point: Offset, val color: Color) : RetouchDepthAction
    data class StrokeContinued(val point: Offset) : RetouchDepthAction
    data object StrokeFinished : RetouchDepthAction
    data class PointerMoved(val point: Offset?) : RetouchDepthAction
    data object ToggleToolsClicked : RetouchDepthAction
    data class TransformChanged(val zoomDelta: Float, val panDelta: Offset) : RetouchDepthAction
    data object ResetTransformClicked : RetouchDepthAction

    data class ApplyClicked(val onResult: (Bitmap) -> Unit) : RetouchDepthAction
    data object ExitRequested : RetouchDepthAction
    data object UndoClicked : RetouchDepthAction
    data object RedoClicked : RetouchDepthAction
    data object CancelClicked : RetouchDepthAction
    data object ClearSelectionClicked : RetouchDepthAction
    data object FillSelectionClicked : RetouchDepthAction
    data object ConfirmTargetDepthClicked : RetouchDepthAction
    data object EyedropperClicked : RetouchDepthAction
}

class RetouchDepthViewModel : ViewModel() {

    private data class SelectionSnapshot(
        val selectionMask: Bitmap?,
        val targetDepth: Int
    )

    private val _uiState = MutableStateFlow(
        RetouchDepthUiState(
            canvasStatus = "Loading...",
            autoSegmentPrompt = "Tap an area to select it, then tap again to pick the target depth"
        )
    )
    val uiState: StateFlow<RetouchDepthUiState> = _uiState.asStateFlow()

    private var currentStrokePoints = mutableListOf<Offset>()
    private val selectionUndoStack = mutableListOf<SelectionSnapshot>()
    private val selectionRedoStack = mutableListOf<SelectionSnapshot>()
    private var interactiveDepthCache: Bitmap? = null
    private var interactiveDepthCacheBase: Bitmap? = null
    private var interactiveDepthCacheStrokes: List<Stroke>? = null
    private var currentDepthArray: FloatArray? = null

    fun onAction(action: RetouchDepthAction) {
        when (action) {
            is RetouchDepthAction.LoadInitialData -> {
                loadData(
                    action.context,
                    action.uri,
                    action.existingDepth,
                    action.forceFreshDepth,
                    action.highQualityDepth
                )
            }
            is RetouchDepthAction.BlurPreviewDefaultsLoaded -> {
                val focus = action.params.focusDepth.toInt().coerceIn(0, 255)
                updateState {
                    it.copy(
                        targetDepth = focus,
                        canvasStatus = if (it.depthMapBitmap != null) "Ready" else "Loading..."
                    )
                }
            }
            is RetouchDepthAction.ModeChanged -> {
                updateState {
                    it.smartSelectPreviewMask?.recycle()
                    it.copy(
                        mode = action.mode,
                        targetDepthConfirmed = false,
                        smartSelectPreviewMask = null,
                        lastSmartSelectPoint = null,
                        canvasStatus = when(action.mode) {
                            RetouchMode.ManualPaint -> "Draw ready"
                            RetouchMode.Erase -> "Erase ready"
                            RetouchMode.GradientBrush -> "Gradient brush ready"
                            RetouchMode.SmartSelect -> "Pick target, tap area, confirm, then fill"
                            RetouchMode.AutoSegment -> "Auto segment ready"
                            RetouchMode.Eyedropper -> "Picker: tap to set target depth"
                        }
                    )
                }
            }
            is RetouchDepthAction.GradientStartChanged -> {
                updateEditedState { it.copy(gradientStartDepth = action.value.coerceIn(0, 255)) }
            }
            is RetouchDepthAction.GradientEndChanged -> {
                updateEditedState { it.copy(gradientEndDepth = action.value.coerceIn(0, 255)) }
            }
            is RetouchDepthAction.ImagePointClicked -> {
                handleImagePointClick(action.imageX, action.imageY)
            }
            is RetouchDepthAction.PointerMoved -> {
                updateState { it.copy(lastTouchPoint = action.point) }
            }
            RetouchDepthAction.ToggleToolsClicked -> {
                updateState { it.copy(showTools = !it.showTools) }
            }
            is RetouchDepthAction.TransformChanged -> {
                updateState {
                    val newScale = (it.zoomScale * action.zoomDelta).coerceIn(1f, 5f)
                    it.copy(zoomScale = newScale, panOffset = it.panOffset + action.panDelta)
                }
            }
            RetouchDepthAction.ResetTransformClicked -> {
                updateState { it.copy(zoomScale = 1f, panOffset = Offset.Zero) }
            }
            is RetouchDepthAction.TargetDepthChanged -> {
                updateEditedState {
                    it.copy(
                        targetDepth = action.value.coerceIn(0, 255),
                        targetDepthConfirmed = false
                    )
                }
            }
            is RetouchDepthAction.DepthOverlayToggled -> {
                updateState { it.copy(showDepthOverlay = action.enabled) }
            }
            is RetouchDepthAction.TargetPickerToggled -> {
                updateState {
                    it.copy(
                        targetPickerEnabled = action.enabled,
                        canvasStatus = when {
                            action.enabled -> "Picker on: tap image to sample target depth"
                            it.mode == RetouchMode.SmartSelect -> "Tap area, confirm, then fill"
                            it.mode == RetouchMode.ManualPaint -> "Draw ready"
                            it.mode == RetouchMode.Erase -> "Erase ready"
                            it.mode == RetouchMode.GradientBrush -> "Gradient brush ready"
                            else -> "Ready"
                        }
                    )
                }
            }
            is RetouchDepthAction.AutoThresholdChanged -> {
                updateEditedState { it.copy(autoSegmentThreshold = action.value.coerceIn(0.01f, 0.15f)) }
                refreshSmartSelectPreview()
            }
            RetouchDepthAction.ConfirmTargetDepthClicked -> {
                confirmSmartSelection()
            }
            
            is RetouchDepthAction.StrokeStarted -> {
                clearInteractiveDepthCache()
                currentStrokePoints = mutableListOf(action.point)
                val currentState = _uiState.value
                val isGradient = currentState.mode == RetouchMode.GradientBrush
                val isErase = currentState.mode == RetouchMode.Erase
                val gray = (currentState.targetDepth / 255f).coerceIn(0f, 1f)
                val paintColor = if (isErase) {
                    Color.White.copy(alpha = 0.28f)
                } else if (currentState.mode == RetouchMode.ManualPaint || isGradient) {
                    Color(gray, gray, gray, 1f)
                } else {
                    action.color
                }

                val newStroke = Stroke(
                    points = currentStrokePoints.toList(),
                    size = currentState.manualBrushSize,
                    color = paintColor,
                    opacity = 1f,
                    isErase = isErase,
                    isGradient = isGradient,
                    gradientStartDepth = currentState.gradientStartDepth,
                    gradientEndDepth = currentState.gradientEndDepth
                )
                updateState { s ->
                    s.copy(
                        strokes = s.strokes + newStroke,
                        undoneStrokes = emptyList(),
                        undoAvailable = true,
                        redoAvailable = false,
                        canvasStatus = if (isErase) "Erasing..." else "Painting..."
                    )
                }
            }
            is RetouchDepthAction.StrokeContinued -> {
                clearInteractiveDepthCache()
                currentStrokePoints.add(action.point)
                updateState { state ->
                    if (state.strokes.isEmpty()) return@updateState state
                    val updatedStroke = state.strokes.last().copy(points = currentStrokePoints.toList())
                    state.copy(strokes = state.strokes.dropLast(1) + updatedStroke)
                }
            }
            RetouchDepthAction.StrokeFinished -> {
                clearInteractiveDepthCache()
                updateState { it.copy(canvasStatus = "Stroke applied", lastTouchPoint = null) }
            }

            RetouchDepthAction.UndoClicked -> {
                val state = _uiState.value
                if (state.strokes.isNotEmpty()) {
                    clearInteractiveDepthCache()
                    updateState { current ->
                        val lastStroke = current.strokes.last()
                        val newStrokes = current.strokes.dropLast(1)
                        val newUndone = current.undoneStrokes + lastStroke
                        current.copy(strokes = newStrokes, undoneStrokes = newUndone)
                    }
                    syncUndoRedoAvailability()
                } else if (selectionUndoStack.isNotEmpty()) {
                    undoSelectionStep()
                }
            }
            RetouchDepthAction.RedoClicked -> {
                val state = _uiState.value
                if (state.undoneStrokes.isNotEmpty()) {
                    clearInteractiveDepthCache()
                    updateState { current ->
                        val strokeToRestore = current.undoneStrokes.last()
                        val newUndone = current.undoneStrokes.dropLast(1)
                        val newStrokes = current.strokes + strokeToRestore
                        current.copy(strokes = newStrokes, undoneStrokes = newUndone)
                    }
                    syncUndoRedoAvailability()
                } else if (selectionRedoStack.isNotEmpty()) {
                    redoSelectionStep()
                }
            }
            RetouchDepthAction.ClearSelectionClicked -> {
                updateState {
                    it.smartSelectPreviewMask?.recycle()
                    it.copy(
                        selectionMask = null,
                        smartSelectPreviewMask = null,
                        lastSmartSelectPoint = null,
                        targetDepthConfirmed = false,
                        canvasStatus = "Selection cleared"
                    )
                }
            }
            RetouchDepthAction.FillSelectionClicked -> {
                fillSelection()
            }
            RetouchDepthAction.EyedropperClicked -> {
                updateState { it.copy(mode = RetouchMode.Eyedropper, targetDepthConfirmed = false, canvasStatus = "Picker active") }
            }
            is RetouchDepthAction.ApplyClicked -> {
                bakeAndApply(action.onResult)
            }
            RetouchDepthAction.ExitRequested -> {
                releaseSessionBitmaps()
            }
            RetouchDepthAction.CancelClicked -> {
                clearSelectionHistory()
                clearInteractiveDepthCache()
                updateState {
                    it.copy(
                        canvasStatus = "Canceled",
                        strokes = emptyList(),
                        undoneStrokes = emptyList(),
                        selectionMask = null,
                        undoAvailable = false,
                        redoAvailable = false
                    )
                }
            }
            is RetouchDepthAction.ManualBrushSizeChanged -> {
                updateEditedState { it.copy(manualBrushSize = action.value) }
            }
            is RetouchDepthAction.FeatherRadiusChanged -> {
                updateEditedState { it.copy(featherRadiusPx = action.value.coerceIn(2f, 36f)) }
            }
            is RetouchDepthAction.EdgeSnapToggled,
            is RetouchDepthAction.ManualBrushHardnessChanged,
            is RetouchDepthAction.ManualBrushOpacityChanged,
            is RetouchDepthAction.AutoMaskPreviewToggled,
            RetouchDepthAction.AutoSegmentDetectRequested -> {
                // Reserved for future use
            }
        }
    }

    private fun loadData(
        context: Context,
        uri: Uri,
        existingDepth: Bitmap?,
        forceFreshDepth: Boolean,
        highQualityDepth: Boolean
    ) {
        viewModelScope.launch {
            try {
                releaseSessionBitmaps()
                clearSelectionHistory()

                // Fast path: use cached bitmaps from preview screen if available
                val cacheMatches = DepthBlurEngine.cachedPreviewUriString == uri.toString() &&
                    DepthBlurEngine.cachedPreviewHighQualityDepth == highQualityDepth
                val cachedSource = if (cacheMatches) {
                    DepthBlurEngine.cachedPreviewSource
                } else {
                    null
                }
                val cachedDepth = if (cacheMatches) {
                    DepthBlurEngine.cachedPreviewDepth
                } else {
                    null
                }
                val depthToUse = existingDepth ?: DepthBlurEngine.getPreferredDepthOverride()

                if (forceFreshDepth && cachedSource != null && cachedDepth != null && existingDepth == null) {
                    val fastSource = cachedSource.copy(Bitmap.Config.ARGB_8888, false)
                    val fastDepth = cachedDepth.copy(Bitmap.Config.ARGB_8888, true)
                    updateState {
                        it.copy(
                            sourceBitmap = fastSource,
                            depthMapBitmap = fastDepth,
                            rawDepthData = null,
                            selectionMask = null,
                            canvasStatus = "Refreshing depth..."
                        )
                    }
                }

                if (!forceFreshDepth && cachedSource != null && (depthToUse != null || cachedDepth != null)) {
                    val srcCopy = cachedSource.copy(Bitmap.Config.ARGB_8888, false)
                    val depthCopy = (depthToUse ?: cachedDepth)!!.copy(Bitmap.Config.ARGB_8888, true)
                    updateState {
                        it.copy(
                            sourceBitmap = srcCopy,
                            depthMapBitmap = depthCopy,
                            rawDepthData = null,
                            selectionMask = null,
                            canvasStatus = "Ready"
                        )
                    }
                    syncUndoRedoAvailability()
                    return@launch
                }

                // Slow path: full pipeline (first time only)
                val (result, depthData) = withContext(Dispatchers.Default) {
                    val res = DepthBlurEngine.renderPreview(
                        context,
                        uri,
                        BlurPreviewParams(),
                        maxDimension = if (highQualityDepth) 960 else 600,
                        overrideDepth = depthToUse,
                        highQualityDepth = highQualityDepth,
                        updateCache = !forceFreshDepth
                    )
                    val raw = if (depthToUse != null) {
                        null
                    } else {
                        val estimator = DepthBlurEngine.getEstimator(context)
                        estimator.estimateDepth(res.sourceBitmap)
                    }
                    res to raw
                }

                val finalDepthMap = if (existingDepth != null) existingDepth.copy(Bitmap.Config.ARGB_8888, true) else result.depthMapBitmap
                currentDepthArray = extractDepthFromBitmap(finalDepthMap)

                updateState {
                    it.copy(
                        sourceBitmap = result.sourceBitmap,
                        depthMapBitmap = finalDepthMap,
                        rawDepthData = if (existingDepth != null) null else depthData,
                        selectionMask = null,
                        canvasStatus = "Ready"
                    )
                }
                clearInteractiveDepthCache()
                syncUndoRedoAvailability()
            } catch (e: Exception) {
                updateState { it.copy(canvasStatus = "Error: ${e.message}") }
            }
        }
    }

    private fun handleImagePointClick(imageX: Int, imageY: Int) {
        val state = _uiState.value
        val depthBitmap = buildInteractiveDepthBitmap(state) ?: return
        val imgX = imageX.coerceIn(0, depthBitmap.width - 1)
        val imgY = imageY.coerceIn(0, depthBitmap.height - 1)

        if (state.targetPickerEnabled) {
            val pixel = depthBitmap.getPixel(imgX, imgY)
            val targetDepthInt = android.graphics.Color.red(pixel)
            updateState {
                it.copy(
                    targetDepth = targetDepthInt,
                    targetPickerEnabled = false,
                    canvasStatus = "Depth sampled: $targetDepthInt"
                )
            }
            releaseInteractiveDepthBitmapIfNeeded(depthBitmap, state)
            return
        }
        
        if (state.mode == RetouchMode.SmartSelect) {
            updateState {
                it.copy(
                    lastSmartSelectPoint = Offset(imgX.toFloat(), imgY.toFloat()),
                    targetDepthConfirmed = false,
                    canvasStatus = "Area previewed. Confirm to add, then fill."
                )
            }
            refreshSmartSelectPreview()
        } else if (state.mode == RetouchMode.Eyedropper) {
            val pixel = depthBitmap.getPixel(imgX, imgY)
            val targetDepthInt = android.graphics.Color.red(pixel)
            updateState {
                it.copy(
                    targetDepth = targetDepthInt,
                    targetDepthConfirmed = false,
                    mode = RetouchMode.SmartSelect,
                    canvasStatus = "Target picked: $targetDepthInt. Tap area next."
                )
            }
        } else {
            val pixel = depthBitmap.getPixel(imgX, imgY)
            val targetDepthInt = android.graphics.Color.red(pixel)
            updateState { it.copy(targetDepth = targetDepthInt) }
        }

        releaseInteractiveDepthBitmapIfNeeded(depthBitmap, state)
    }

    private fun refreshSmartSelectPreview() {
        val state = _uiState.value
        val point = state.lastSmartSelectPoint ?: return
        val depthBitmap = buildInteractiveDepthBitmap(state) ?: return
        val bitmap = state.sourceBitmap ?: return

        viewModelScope.launch {
            val previewMask = withContext(Dispatchers.Default) {
                val currentDepthData = extractDepthFromBitmap(depthBitmap)
                val mask = DepthBlurEngine.findSimilarDepthRegion(
                    currentDepthData,
                    bitmap.width,
                    bitmap.height,
                    point.x.toInt(),
                    point.y.toInt(),
                    state.autoSegmentThreshold
                )
                // Use a different color for preview (e.g. orange-ish)
                DepthBlurEngine.maskToBitmap(mask, bitmap.width, bitmap.height, 0x99FFC145.toInt())
            }
            updateState {
                it.smartSelectPreviewMask?.recycle()
                it.copy(smartSelectPreviewMask = previewMask)
            }
            releaseInteractiveDepthBitmapIfNeeded(depthBitmap, state)
        }
    }

    private fun fillSelection() {
        val state = _uiState.value
        val selectionMask = state.selectionMask ?: return
        val depthBitmap = buildInteractiveDepthBitmap(state) ?: return
        
        // Safety check to prevent crashes if dimensions mismatch
        if (selectionMask.width != depthBitmap.width || selectionMask.height != depthBitmap.height) {
            updateState { it.copy(canvasStatus = "Selection size mismatch") }
            return
        }

        viewModelScope.launch {
            updateState { it.copy(autoSegmentBusy = true, canvasStatus = "Filling selection...") }
            
            val newDepth = withContext(Dispatchers.Default) {
                val baked = depthBitmap.copy(Bitmap.Config.ARGB_8888, true)
                val width = baked.width
                val height = baked.height
                
                val maskPixels = IntArray(width * height)
                selectionMask.getPixels(maskPixels, 0, width, 0, 0, width, height)
                val bakedPixels = IntArray(width * height)
                baked.getPixels(bakedPixels, 0, width, 0, 0, width, height)
                
                val targetGray = state.targetDepth
                val targetColor = android.graphics.Color.rgb(targetGray, targetGray, targetGray) or (0xFF shl 24)

                for (i in maskPixels.indices) {
                    if (android.graphics.Color.alpha(maskPixels[i]) > 10) {
                        bakedPixels[i] = targetColor
                    }
                }
                val selectionMaskFloat = FloatArray(maskPixels.size) { index ->
                    android.graphics.Color.alpha(maskPixels[index]) / 255f
                }
                homogenizeEditedRegion(
                    editedPixels = bakedPixels,
                    originalPixels = null,
                    editMask = selectionMaskFloat,
                    width = width,
                    height = height,
                    radius = 5,
                    strength = 0.55f
                )
                baked.setPixels(bakedPixels, 0, width, 0, 0, width, height)
                baked
            }

            pushSelectionUndoSnapshot(state)
            updateState {
                it.copy(
                    depthMapBitmap = newDepth,
                    selectionMask = null,
                    smartSelectPreviewMask = null,
                    strokes = emptyList(),
                    undoneStrokes = emptyList(),
                    targetDepthConfirmed = false,
                    autoSegmentBusy = false,
                    canvasStatus = "Selection filled"
                )
            }
            syncUndoRedoAvailability()
            clearInteractiveDepthCache()
        }
    }

    private fun confirmSmartSelection() {
        val state = _uiState.value
        val previewMask = state.smartSelectPreviewMask
        if (previewMask == null) {
            updateState { it.copy(canvasStatus = "Select an area first") }
            return
        }

        pushSelectionUndoSnapshot(state)
        val confirmedMask = recolorMask(previewMask, 0x995CD6B3.toInt())
        val combinedMask = if (state.selectionMask != null) {
            val union = state.selectionMask.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(union)
            canvas.drawBitmap(confirmedMask, 0f, 0f, null)
            confirmedMask.recycle()
            union
        } else {
            confirmedMask
        }

        updateState {
            it.smartSelectPreviewMask?.recycle()
            it.copy(
                selectionMask = combinedMask,
                smartSelectPreviewMask = null,
                targetDepthConfirmed = true,
                canvasStatus = "Selection confirmed. Fill to apply."
            )
        }
        syncUndoRedoAvailability()
    }

    private fun buildInteractiveDepthBitmap(state: RetouchDepthUiState): Bitmap? {
        val depthBitmap = state.depthMapBitmap ?: return null
        if (state.strokes.isEmpty()) {
            clearInteractiveDepthCache()
            return depthBitmap
        }
        interactiveDepthCache?.let { cached ->
            if (
                !cached.isRecycled &&
                interactiveDepthCacheBase === depthBitmap &&
                interactiveDepthCacheStrokes === state.strokes
            ) {
                return cached
            }
        }
        clearInteractiveDepthCache()

        val baked = depthBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(baked)
        val width = baked.width
        val height = baked.height
        val originalPixels = IntArray(width * height).also {
            depthBitmap.getPixels(it, 0, width, 0, 0, width, height)
        }

        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
        }

        for (stroke in state.strokes) {
            if (stroke.isErase) continue
            paint.strokeWidth = stroke.size
            paint.color = stroke.color.toArgb()
            val path = Path()
            if (stroke.points.isNotEmpty()) {
                path.moveTo(stroke.points[0].x, stroke.points[0].y)
                for (i in 1 until stroke.points.size) {
                    path.lineTo(stroke.points[i].x, stroke.points[i].y)
                }
                canvas.drawPath(path, paint)
            }
            if (stroke.points.size == 1) {
                val point = stroke.points.first()
                canvas.drawCircle(point.x, point.y, stroke.size / 2f, paint)
            }
        }

        val bakedPixels = IntArray(width * height)
        baked.getPixels(bakedPixels, 0, width, 0, 0, width, height)

        for (stroke in state.strokes) {
            if (!stroke.isErase) continue
            val strokeMaskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val strokeMaskCanvas = Canvas(strokeMaskBitmap)
            val strokeMaskPaint = Paint().apply {
                color = android.graphics.Color.WHITE
                style = Paint.Style.STROKE
                strokeWidth = stroke.size
                strokeCap = Paint.Cap.ROUND
                strokeJoin = Paint.Join.ROUND
                isAntiAlias = true
            }

            if (stroke.points.isNotEmpty()) {
                val path = Path().apply {
                    moveTo(stroke.points[0].x, stroke.points[0].y)
                    for (i in 1 until stroke.points.size) {
                        lineTo(stroke.points[i].x, stroke.points[i].y)
                    }
                }
                strokeMaskCanvas.drawPath(path, strokeMaskPaint)
            }
            if (stroke.points.size == 1) {
                val point = stroke.points.first()
                strokeMaskCanvas.drawCircle(point.x, point.y, stroke.size / 2f, strokeMaskPaint)
            }

            val strokeMask = extractAlphaMask(strokeMaskBitmap)
            strokeMaskBitmap.recycle()
            for (i in bakedPixels.indices) {
                if (strokeMask[i] > 0f) {
                    bakedPixels[i] = originalPixels[i]
                }
            }
        }

        baked.setPixels(bakedPixels, 0, width, 0, 0, width, height)
        interactiveDepthCache = baked
        interactiveDepthCacheBase = depthBitmap
        interactiveDepthCacheStrokes = state.strokes
        return baked
    }

    private fun recolorMask(maskBitmap: Bitmap, color: Int): Bitmap {
        val width = maskBitmap.width
        val height = maskBitmap.height
        val pixels = IntArray(width * height)
        maskBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        for (i in pixels.indices) {
            pixels[i] = if (android.graphics.Color.alpha(pixels[i]) > 10) color else 0
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun releaseInteractiveDepthBitmapIfNeeded(bitmap: Bitmap, state: RetouchDepthUiState) {
        if (bitmap !== state.depthMapBitmap && bitmap !== interactiveDepthCache) {
            bitmap.recycle()
        }
    }

    private fun clearInteractiveDepthCache() {
        interactiveDepthCache?.recycle()
        interactiveDepthCache = null
        interactiveDepthCacheBase = null
        interactiveDepthCacheStrokes = null
    }

    private fun bakeAndApply(onResult: (Bitmap) -> Unit) {
        val state = _uiState.value
        val depthBitmap = state.depthMapBitmap ?: return
        val strokes = state.strokes
        val width = depthBitmap.width
        val height = depthBitmap.height

        viewModelScope.launch {
            updateState { it.copy(autoSegmentBusy = true, canvasStatus = "Saving...") }
            
            val finalDepth = withContext(Dispatchers.Default) {
                val baked = depthBitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(baked)
                val originalPixels = IntArray(width * height).also {
                    depthBitmap.getPixels(it, 0, width, 0, 0, width, height)
                }
                val editMaskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                val editMaskCanvas = Canvas(editMaskBitmap)
                val editMaskPaint = Paint().apply {
                    color = android.graphics.Color.WHITE
                    style = Paint.Style.STROKE
                    strokeCap = Paint.Cap.ROUND
                    strokeJoin = Paint.Join.ROUND
                    isAntiAlias = true
                }
                
                // 1. Bake non-erase strokes
                val paint = Paint().apply {
                    style = Paint.Style.STROKE
                    strokeCap = Paint.Cap.ROUND
                    strokeJoin = Paint.Join.ROUND
                    isAntiAlias = true
                }

                for (stroke in strokes) {
                    editMaskPaint.strokeWidth = stroke.size
                    val path = Path()
                    if (stroke.points.isNotEmpty()) {
                        path.moveTo(stroke.points[0].x, stroke.points[0].y)
                        for (i in 1 until stroke.points.size) {
                            path.lineTo(stroke.points[i].x, stroke.points[i].y)
                        }
                        editMaskCanvas.drawPath(path, editMaskPaint)
                        if (!stroke.isErase) {
                            paint.strokeWidth = stroke.size
                            paint.color = stroke.color.toArgb()
                            canvas.drawPath(path, paint)
                        }
                    }
                    if (stroke.points.size == 1) {
                        val point = stroke.points.first()
                        editMaskCanvas.drawCircle(point.x, point.y, stroke.size / 2f, editMaskPaint)
                        if (!stroke.isErase) {
                            paint.strokeWidth = stroke.size
                            paint.color = stroke.color.toArgb()
                            canvas.drawCircle(point.x, point.y, stroke.size / 2f, paint)
                        }
                    }
                }

                val bakedPixels = IntArray(width * height)
                baked.getPixels(bakedPixels, 0, width, 0, 0, width, height)

                // 3. Restore original depth inside erase strokes
                for (stroke in strokes) {
                    if (!stroke.isErase) continue
                    val strokeMaskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                    val strokeMaskCanvas = Canvas(strokeMaskBitmap)
                    val strokeMaskPaint = Paint().apply {
                        color = android.graphics.Color.WHITE
                        style = Paint.Style.STROKE
                        strokeWidth = stroke.size
                        strokeCap = Paint.Cap.ROUND
                        strokeJoin = Paint.Join.ROUND
                        isAntiAlias = true
                    }

                    if (stroke.points.isNotEmpty()) {
                        val path = Path().apply {
                            moveTo(stroke.points[0].x, stroke.points[0].y)
                            for (i in 1 until stroke.points.size) {
                                lineTo(stroke.points[i].x, stroke.points[i].y)
                            }
                        }
                        strokeMaskCanvas.drawPath(path, strokeMaskPaint)
                    }
                    if (stroke.points.size == 1) {
                        val point = stroke.points.first()
                        strokeMaskCanvas.drawCircle(point.x, point.y, stroke.size / 2f, strokeMaskPaint)
                    }

                    val strokeMask = extractAlphaMask(strokeMaskBitmap)
                    strokeMaskBitmap.recycle()
                    for (i in bakedPixels.indices) {
                        if (strokeMask[i] > 0f) {
                            bakedPixels[i] = originalPixels[i]
                        }
                    }
                }

                val editMask = extractAlphaMask(editMaskBitmap)
                editMaskBitmap.recycle()
                val featherRadius = state.featherRadiusPx.toInt().coerceIn(1, 18)
                val featheredMask = blurMask(editMask, width, height, featherRadius)

                // Safety check: ensure arrays match
                if (featheredMask.size == bakedPixels.size && originalPixels.size == bakedPixels.size) {
                    for (i in bakedPixels.indices) {
                        val blend = featheredMask[i].coerceIn(0f, 1f)
                        if (blend <= 0f) continue

                        val originalGray = android.graphics.Color.red(originalPixels[i])
                        val editedGray = android.graphics.Color.red(bakedPixels[i])
                        val finalGray = (originalGray + (editedGray - originalGray).toFloat() * blend).toInt().coerceIn(0, 255)
                        bakedPixels[i] = android.graphics.Color.argb(255, finalGray, finalGray, finalGray)
                    }
                }

                homogenizeEditedRegion(
                    editedPixels = bakedPixels,
                    originalPixels = originalPixels,
                    editMask = featheredMask,
                    width = width,
                    height = height,
                    radius = (featherRadius / 2).coerceIn(2, 8),
                    strength = 0.42f
                )
                baked.setPixels(bakedPixels, 0, width, 0, 0, width, height)
                baked
            }
            
            onResult(finalDepth)
            releaseSessionBitmaps()
        }
    }

    private fun extractDepthFromBitmap(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val depth = FloatArray(w * h)
        for (i in pixels.indices) depth[i] = android.graphics.Color.red(pixels[i]) / 255f
        return depth
    }

    private fun extractAlphaMask(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        return FloatArray(w * h) { index ->
            android.graphics.Color.alpha(pixels[index]) / 255f
        }
    }

    private fun blurMask(mask: FloatArray, width: Int, height: Int, radius: Int): FloatArray {
        if (radius <= 0) return mask
        val temp = FloatArray(mask.size)
        val out = FloatArray(mask.size)
        boxBlurHorizontal(mask, temp, width, height, radius)
        boxBlurVertical(temp, out, width, height, radius)
        return out
    }

    private fun homogenizeEditedRegion(
        editedPixels: IntArray,
        originalPixels: IntArray?,
        editMask: FloatArray,
        width: Int,
        height: Int,
        radius: Int,
        strength: Float
    ) {
        if (editedPixels.isEmpty() || editMask.isEmpty()) return
        val clampedStrength = strength.coerceIn(0f, 1f)
        if (clampedStrength <= 0f) return

        val editedGray = FloatArray(editedPixels.size) { index ->
            android.graphics.Color.red(editedPixels[index]).toFloat()
        }
        val blurredGray = blurMask(editedGray, width, height, radius.coerceAtLeast(1))
        val centerMask = blurMask(editMask, width, height, (radius / 2).coerceAtLeast(1))

        for (i in editedPixels.indices) {
            val maskValue = centerMask[i].coerceIn(0f, 1f)
            if (maskValue <= 0.02f) continue

            val interiorWeight = (maskValue * maskValue * clampedStrength).coerceIn(0f, 1f)
            if (interiorWeight <= 0f) continue

            val originalGray = originalPixels?.let { android.graphics.Color.red(it[i]).toFloat() } ?: editedGray[i]
            val targetGray = editedGray[i] + (blurredGray[i] - editedGray[i]) * interiorWeight
            val preservedGray = originalGray + (targetGray - originalGray) * maskValue
            val finalGray = preservedGray.toInt().coerceIn(0, 255)
            editedPixels[i] = android.graphics.Color.argb(255, finalGray, finalGray, finalGray)
        }
    }

    private fun boxBlurHorizontal(src: FloatArray, dst: FloatArray, width: Int, height: Int, radius: Int) {
        val div = (radius * 2 + 1).toFloat()
        for (y in 0 until height) {
            val row = y * width
            var sum = 0f
            for (dx in -radius..radius) {
                sum += src[row + dx.coerceIn(0, width - 1)]
            }
            dst[row] = sum / div
            for (x in 1 until width) {
                sum += src[row + (x + radius).coerceIn(0, width - 1)]
                sum -= src[row + (x - radius - 1).coerceIn(0, width - 1)]
                dst[row + x] = sum / div
            }
        }
    }

    private fun boxBlurVertical(src: FloatArray, dst: FloatArray, width: Int, height: Int, radius: Int) {
        val div = (radius * 2 + 1).toFloat()
        for (x in 0 until width) {
            var sum = 0f
            for (dy in -radius..radius) {
                sum += src[dy.coerceIn(0, height - 1) * width + x]
            }
            dst[x] = sum / div
            for (y in 1 until height) {
                sum += src[(y + radius).coerceIn(0, height - 1) * width + x]
                sum -= src[(y - radius - 1).coerceIn(0, height - 1) * width + x]
                dst[y * width + x] = sum / div
            }
        }
    }

    private fun updateEditedState(reducer: (RetouchDepthUiState) -> RetouchDepthUiState) {
        _uiState.update { current -> reducer(current).copy(canvasStatus = "Edited") }
        syncUndoRedoAvailability()
    }

    private fun updateState(reducer: (RetouchDepthUiState) -> RetouchDepthUiState) {
        _uiState.update(reducer)
    }

    private fun releaseSessionBitmaps() {
        clearInteractiveDepthCache()
        _uiState.update {
            it.sourceBitmap?.recycle()
            it.depthMapBitmap?.recycle()
            it.selectionMask?.recycle()
            it.smartSelectPreviewMask?.recycle()
            it.copy(
                sourceBitmap = null,
                depthMapBitmap = null,
                selectionMask = null,
                smartSelectPreviewMask = null,
                strokes = emptyList(),
                undoneStrokes = emptyList()
            )
        }
        clearSelectionHistory()
    }

    private fun pushSelectionUndoSnapshot(state: RetouchDepthUiState) {
        selectionUndoStack.add(
            SelectionSnapshot(
                selectionMask = copyBitmap(state.selectionMask),
                targetDepth = state.targetDepth
            )
        )
        clearSelectionRedoStack()
    }

    private fun undoSelectionStep() {
        val state = _uiState.value
        val restore = selectionUndoStack.removeLastOrNull() ?: return
        selectionRedoStack.add(
            SelectionSnapshot(
                selectionMask = copyBitmap(state.selectionMask),
                targetDepth = state.targetDepth
            )
        )
        updateState {
            it.copy(
                selectionMask = restore.selectionMask,
                targetDepth = restore.targetDepth,
                canvasStatus = "Selection step undone"
            )
        }
        syncUndoRedoAvailability()
    }

    private fun redoSelectionStep() {
        val state = _uiState.value
        val restore = selectionRedoStack.removeLastOrNull() ?: return
        selectionUndoStack.add(
            SelectionSnapshot(
                selectionMask = copyBitmap(state.selectionMask),
                targetDepth = state.targetDepth
            )
        )
        updateState {
            it.copy(
                selectionMask = restore.selectionMask,
                targetDepth = restore.targetDepth,
                canvasStatus = "Selection step restored"
            )
        }
        syncUndoRedoAvailability()
    }

    private fun syncUndoRedoAvailability() {
        _uiState.update { state ->
            state.copy(
                undoAvailable = state.strokes.isNotEmpty() || selectionUndoStack.isNotEmpty(),
                redoAvailable = state.undoneStrokes.isNotEmpty() || selectionRedoStack.isNotEmpty()
            )
        }
    }

    private fun clearSelectionHistory() {
        selectionUndoStack.forEach { it.selectionMask?.recycle() }
        selectionUndoStack.clear()
        clearSelectionRedoStack()
    }

    private fun clearSelectionRedoStack() {
        selectionRedoStack.forEach { it.selectionMask?.recycle() }
        selectionRedoStack.clear()
    }

    private fun copyBitmap(bitmap: Bitmap?): Bitmap? =
        bitmap?.copy(Bitmap.Config.ARGB_8888, true)

    override fun onCleared() {
        releaseSessionBitmaps()
        super.onCleared()
    }
}
