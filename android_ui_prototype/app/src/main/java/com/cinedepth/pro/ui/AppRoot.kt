package com.cinedepth.pro.ui

import android.graphics.Bitmap
import android.net.Uri
import android.widget.Toast
import android.os.Handler
import android.os.Looper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.outlined.AutoFixHigh
import androidx.compose.material.icons.outlined.BlurOn
import androidx.compose.material.icons.outlined.Layers
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.cinedepth.pro.ui.blur.DepthBlurEngine
import com.cinedepth.pro.ui.blur.DepthBlurExportResult
import com.cinedepth.pro.ui.retouch.RetouchDepthRoute
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.min

// ─── Shared Colors ──────────────────────────────────────────────────────────

private val GlassBlack = Color(0xCC1A1A1A)
private val GlassDark = Color(0xB3000000)
private val AccentAmber = Color(0xFFFFC145)
private val SoftWhite = Color(0xFFF0F0F0)

data class BlurPreviewParams(
    val blurStrength: Float = 0.20f,
    val focusDepth: Float = 128f,
    val lensEffect: LensEffect = LensEffect.Classic,
    val edgeSoftness: Float = 0.35f,
    val edgeExpand: Float = 0.22f,
    val edgeRefine: Float = 0.52f,
    val backgroundLight: Float = 0.12f,
    val highlightBoost: Float = 0.20f,
    val blurFalloff: Float = 0.75f,
    val vignetteStrength: Float = 0.0f,
    val flareStrength: Float = 0.05f
)

enum class LensEffect {
    Classic,
    Creamy,
    Bubble,
    Bloom,
    Star,
    Hexagon,
    Anamorphic
}

enum class BokehPreset(
    val label: String,
    val effect: LensEffect
) {
    Classic("Classic", LensEffect.Classic),
    Creamy("Creamy", LensEffect.Creamy),
    Bubble("Bubble", LensEffect.Bubble),
    Bloom("Bloom", LensEffect.Bloom),
    Star("Star", LensEffect.Star),
    Circular("Circular", LensEffect.Hexagon),
    Anamorphic("Anamorphic", LensEffect.Anamorphic)
}

enum class LensProfile(
    val label: String,
    val params: BlurPreviewParams
) {
    Noctilux(
        label = "Noctilux",
        params = BlurPreviewParams(
            blurStrength = 0.42f,
            lensEffect = LensEffect.Creamy,
            edgeSoftness = 0.28f,
            edgeExpand = 0.18f,
            edgeRefine = 0.60f,
            backgroundLight = 0.18f,
            highlightBoost = 0.32f,
            blurFalloff = 0.55f,
            vignetteStrength = 0.20f,
            flareStrength = 0.08f
        )
    ),
    GMaster(
        label = "G-Master",
        params = BlurPreviewParams(
            blurStrength = 0.34f,
            lensEffect = LensEffect.Classic,
            edgeSoftness = 0.40f,
            edgeExpand = 0.24f,
            edgeRefine = 0.65f,
            backgroundLight = 0.10f,
            highlightBoost = 0.15f,
            blurFalloff = 0.68f,
            vignetteStrength = 0.06f,
            flareStrength = 0.03f
        )
    ),
    Helios(
        label = "Helios 44-2",
        params = BlurPreviewParams(
            blurStrength = 0.38f,
            lensEffect = LensEffect.Bubble,
            edgeSoftness = 0.22f,
            edgeExpand = 0.14f,
            edgeRefine = 0.44f,
            backgroundLight = 0.22f,
            highlightBoost = 0.28f,
            blurFalloff = 0.48f,
            vignetteStrength = 0.26f,
            flareStrength = 0.12f
        )
    ),
    CinemaScope(
        label = "CinemaScope",
        params = BlurPreviewParams(
            blurStrength = 0.36f,
            lensEffect = LensEffect.Anamorphic,
            edgeSoftness = 0.32f,
            edgeExpand = 0.20f,
            edgeRefine = 0.55f,
            backgroundLight = 0.14f,
            highlightBoost = 0.30f,
            blurFalloff = 0.62f,
            vignetteStrength = 0.15f,
            flareStrength = 0.18f
        )
    )
}

private enum class AppScreen {
    Welcome,
    BlurPreview,
    Retouch
}

private enum class PreviewDisplayMode {
    Blur,
    DepthMap
}

private enum class PreviewControlTab {
    Lens,
    Size,
    Edge,
    Light
}

private enum class DepthQualityMode(
    val label: String,
    val description: String,
    val usesHighQualityPath: Boolean
) {
    Standard(
        label = "Standard",
        description = "Fastest on-device depth.",
        usesHighQualityPath = false
    ),
    Hd(
        label = "HD",
        description = "Sharper on-device depth for preview, retouch and save.",
        usesHighQualityPath = true
    ),
    Ultra(
        label = "Ultra",
        description = "Depth Pro coming soon.",
        usesHighQualityPath = true
    )
}

@Composable
fun CineDepthApp() {
    var currentScreen by remember { mutableStateOf(AppScreen.Welcome.name) }
    var selectedPhotoUriString by remember { mutableStateOf<String?>(null) }
    var blurStrength by rememberSaveable { mutableStateOf(0.20f) }
    var focusDepth by rememberSaveable { mutableStateOf(128f) }
    var lensEffectRaw by rememberSaveable { mutableStateOf(LensEffect.Classic.name) }
    var edgeSoftness by rememberSaveable { mutableStateOf(0.35f) }
    var edgeExpand by rememberSaveable { mutableStateOf(0.22f) }
    var edgeRefine by rememberSaveable { mutableFloatStateOf(0.52f) }
    var backgroundLight by rememberSaveable { mutableFloatStateOf(0.12f) }
    var highlightBoost by rememberSaveable { mutableFloatStateOf(0.20f) }
    var blurFalloff by rememberSaveable { mutableFloatStateOf(0.75f) }
    var vignetteStrength by rememberSaveable { mutableFloatStateOf(0.0f) }
    var flareStrength by rememberSaveable { mutableFloatStateOf(0.05f) }
    var depthQualityRaw by rememberSaveable { mutableStateOf(DepthQualityMode.Standard.name) }
    var depthRevision by remember { mutableStateOf(0) }
    var retouchFreshDepthKey by remember { mutableStateOf(0) }
    val depthQuality = remember(depthQualityRaw) {
        when (val parsed = runCatching { DepthQualityMode.valueOf(depthQualityRaw) }.getOrDefault(DepthQualityMode.Standard)) {
            DepthQualityMode.Ultra -> DepthQualityMode.Hd
            else -> parsed
        }
    }

    val blurPreviewParams = BlurPreviewParams(
        blurStrength = blurStrength,
        focusDepth = focusDepth,
        lensEffect = LensEffect.valueOf(lensEffectRaw),
        edgeSoftness = edgeSoftness,
        edgeExpand = edgeExpand,
        edgeRefine = edgeRefine,
        backgroundLight = backgroundLight,
        highlightBoost = highlightBoost,
        blurFalloff = blurFalloff,
        vignetteStrength = vignetteStrength,
        flareStrength = flareStrength
    )

    LaunchedEffect(selectedPhotoUriString) {
        if (selectedPhotoUriString == null) {
            DepthBlurEngine.clearTransientState()
            depthRevision = 0
        }
    }

    LaunchedEffect(currentScreen, selectedPhotoUriString) {
        if (currentScreen == AppScreen.Welcome.name && !selectedPhotoUriString.isNullOrBlank()) {
            currentScreen = AppScreen.BlurPreview.name
        }
    }

    val screen = AppScreen.valueOf(currentScreen)

    val onPhotoSelected: (Uri?) -> Unit = remember { { selectedPhotoUriString = it?.toString() } }
    val onBlurParamsChange: (BlurPreviewParams) -> Unit = remember {
        { params ->
            blurStrength = params.blurStrength
            focusDepth = params.focusDepth
            lensEffectRaw = params.lensEffect.name
            edgeSoftness = params.edgeSoftness
            edgeExpand = params.edgeExpand
            edgeRefine = params.edgeRefine
            backgroundLight = params.backgroundLight
            highlightBoost = params.highlightBoost
            blurFalloff = params.blurFalloff
            vignetteStrength = params.vignetteStrength
            flareStrength = params.flareStrength
        }
    }
    val onBackFromPreview = remember {
        {
            selectedPhotoUriString = null
            currentScreen = AppScreen.Welcome.name
        }
    }
    val onRetouchDepthClick = remember {
        {
            retouchFreshDepthKey = 0
            currentScreen = AppScreen.Retouch.name
        }
    }
    val onApplyRetouch = remember {
        {
            retouchFreshDepthKey = 0
            depthRevision++
            currentScreen = AppScreen.BlurPreview.name
        }
    }
    val onBackFromRetouch = remember {
        {
            retouchFreshDepthKey = 0
            currentScreen = AppScreen.BlurPreview.name
        }
    }

    AnimatedContent(
        targetState = screen,
        transitionSpec = {
            if (targetState.ordinal > initialState.ordinal) {
                (slideInHorizontally { it / 3 } + fadeIn()) togetherWith
                        (slideOutHorizontally { -it / 3 } + fadeOut())
            } else {
                (slideInHorizontally { -it / 3 } + fadeIn()) togetherWith
                        (slideOutHorizontally { it / 3 } + fadeOut())
            }
        },
        label = "screen_transition"
    ) { targetScreen ->
        when (targetScreen) {
            AppScreen.Welcome -> {
                WelcomeScreen(onPhotoSelected = onPhotoSelected)
            }

            AppScreen.BlurPreview -> {
                BlurPreviewScreen(
                    selectedPhotoUriString = selectedPhotoUriString,
                    blurParams = blurPreviewParams,
                    depthQuality = depthQuality,
                    depthRevision = depthRevision,
                    onDepthQualityChange = { quality ->
                        if (depthQuality != quality) {
                            depthQualityRaw = quality.name
                            DepthBlurEngine.clearPreviewCache()
                            depthRevision++
                            retouchFreshDepthKey++
                        }
                    },
                    onBlurParamsChange = onBlurParamsChange,
                    onBack = onBackFromPreview,
                    onRetouchDepthClick = onRetouchDepthClick
                )
            }

            AppScreen.Retouch -> {
                RetouchDepthRoute(
                    selectedPhotoUriString = selectedPhotoUriString,
                    blurPreviewParams = blurPreviewParams,
                    highQualityDepthEnabled = depthQuality.usesHighQualityPath,
                    existingDepth = activeDepthOverrideForQuality()?.copy(Bitmap.Config.ARGB_8888, true),
                    forceFreshDepth = retouchFreshDepthKey > 0,
                    onApplyRetouch = onApplyRetouch,
                    onBack = onBackFromRetouch
                )
            }
        }
    }
}

// ─── Welcome Screen ──────────────────────────────────────────────────────────

@Composable
private fun WelcomeScreen(
    onPhotoSelected: (Uri?) -> Unit
) {
    val pickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri -> onPhotoSelected(uri) }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color(0xFF0D0D0D)
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(WindowInsets.statusBars.asPaddingValues())
        ) {
            // Subtle gradient background
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        Brush.verticalGradient(
                            colors = listOf(
                                Color(0xFF1A1400),
                                Color(0xFF0D0D0D),
                                Color(0xFF0D0D0D)
                            )
                        )
                    )
            )

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 32.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.weight(1f))

                AppLogoBadge()

                Spacer(modifier = Modifier.height(32.dp))

                Text(
                    text = "Real Depth Photo",
                    style = MaterialTheme.typography.headlineLarge,
                    color = SoftWhite,
                    textAlign = TextAlign.Center
                )

                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    text = "AI-powered depth of field\nfor any photo",
                    style = MaterialTheme.typography.bodyLarge,
                    color = Color.White.copy(alpha = 0.5f),
                    textAlign = TextAlign.Center,
                    lineHeight = 26.sp
                )

                Spacer(modifier = Modifier.height(48.dp))

                // Feature chips
                Column(
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    horizontalAlignment = Alignment.Start,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    FeatureChip(icon = Icons.Outlined.AutoFixHigh, text = "AI Depth Estimation")
                    FeatureChip(icon = Icons.Outlined.BlurOn, text = "Realistic Bokeh & Lens Effects")
                    FeatureChip(icon = Icons.Outlined.Layers, text = "Manual Depth Editing")
                }

                Spacer(modifier = Modifier.weight(1f))

                Button(
                    onClick = {
                        pickerLauncher.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(58.dp),
                    shape = RoundedCornerShape(16.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = AccentAmber,
                        contentColor = Color(0xFF1A1200)
                    ),
                    contentPadding = PaddingValues(horizontal = 24.dp)
                ) {
                    Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(22.dp))
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        "Select Photo",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    text = "Supports JPEG, PNG, HEIC",
                    style = MaterialTheme.typography.labelMedium,
                    color = Color.White.copy(alpha = 0.3f)
                )

                Spacer(modifier = Modifier.height(
                    WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding() + 24.dp
                ))
            }
        }
    }
}

@Composable
private fun AppLogoBadge(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .size(104.dp)
            .clip(RoundedCornerShape(30.dp))
            .background(
                Brush.linearGradient(
                    colors = listOf(
                        Color(0xFF241700),
                        Color(0xFF100E0A)
                    )
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        Canvas(
            modifier = Modifier
                .size(84.dp)
                .padding(8.dp)
        ) {
            val stroke = min(size.width, size.height) * 0.12f
            val inset = stroke * 0.85f
            val frameColor = Color(0xFFFFD27A)
            val focusColor = Color(0xFFFFA21A)

            drawRoundRect(
                brush = Brush.linearGradient(
                    colors = listOf(frameColor, focusColor),
                    start = Offset.Zero,
                    end = Offset(size.width, size.height)
                ),
                cornerRadius = CornerRadius(size.minDimension * 0.24f, size.minDimension * 0.24f)
            )

            drawRoundRect(
                color = Color(0xFF120F0B),
                topLeft = Offset(inset, inset),
                size = Size(size.width - inset * 2f, size.height - inset * 2f),
                cornerRadius = CornerRadius(size.minDimension * 0.18f, size.minDimension * 0.18f)
            )

            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(Color(0xFFFFC14C), Color(0xFF7D4700)),
                    center = center,
                    radius = size.minDimension * 0.34f
                ),
                radius = size.minDimension * 0.28f,
                center = center
            )

            drawCircle(
                color = Color(0xFF14100A),
                radius = size.minDimension * 0.16f,
                center = center
            )

            drawArc(
                color = frameColor.copy(alpha = 0.92f),
                startAngle = 220f,
                sweepAngle = 100f,
                useCenter = false,
                topLeft = Offset(size.width * 0.2f, size.height * 0.2f),
                size = Size(size.width * 0.6f, size.height * 0.6f),
                style = Stroke(
                    width = stroke * 0.48f,
                    cap = androidx.compose.ui.graphics.StrokeCap.Round,
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(stroke * 0.8f, stroke * 0.9f))
                )
            )
        }
    }
}

@Composable
private fun FeatureChip(icon: ImageVector, text: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Color.White.copy(alpha = 0.06f))
            .padding(horizontal = 16.dp, vertical = 12.dp)
            .fillMaxWidth()
    ) {
        Icon(
            icon,
            contentDescription = null,
            tint = AccentAmber,
            modifier = Modifier.size(20.dp)
        )
        Spacer(modifier = Modifier.width(14.dp))
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = Color.White.copy(alpha = 0.8f)
        )
    }
}

// ─── Blur Preview Screen (Full-Screen Immersive) ─────────────────────────────

@Composable
private fun BlurPreviewScreen(
    selectedPhotoUriString: String?,
    blurParams: BlurPreviewParams,
    depthQuality: DepthQualityMode,
    depthRevision: Int,
    onDepthQualityChange: (DepthQualityMode) -> Unit,
    onBlurParamsChange: (BlurPreviewParams) -> Unit,
    onBack: () -> Unit,
    onRetouchDepthClick: () -> Unit
) {
    var saveStatus by rememberSaveable { mutableStateOf<String?>(null) }
    var saveDialogVisible by rememberSaveable { mutableStateOf(false) }
    var isSaving by rememberSaveable { mutableStateOf(false) }
    var saveMenuExpanded by remember { mutableStateOf(false) }
    var previewDisplayModeRaw by rememberSaveable { mutableStateOf(PreviewDisplayMode.Blur.name) }
    var activeControlTabRaw by rememberSaveable { mutableStateOf(PreviewControlTab.Lens.name) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val previewDisplayMode = remember(previewDisplayModeRaw) { PreviewDisplayMode.valueOf(previewDisplayModeRaw) }
    val activeControlTab = remember(activeControlTabRaw) { PreviewControlTab.valueOf(activeControlTabRaw) }
    val statusBarPadding = WindowInsets.statusBars.asPaddingValues()
    val navBarPadding = WindowInsets.navigationBars.asPaddingValues()
    var focusPointX by rememberSaveable { mutableFloatStateOf(0.5f) }
    var focusPointY by rememberSaveable { mutableFloatStateOf(0.42f) }
    val applyBlurParamsChange: (BlurPreviewParams) -> Unit = { params ->
        previewDisplayModeRaw = PreviewDisplayMode.Blur.name
        onBlurParamsChange(params)
    }

    if (saveDialogVisible && saveStatus != null) {
        AlertDialog(
            onDismissRequest = { saveDialogVisible = false },
            confirmButton = {
                TextButton(onClick = { saveDialogVisible = false }) { Text("Done") }
            },
            title = { Text(if (saveStatus!!.startsWith("Save failed")) "Error" else "Saved") },
            text = { Text(saveStatus!!) }
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // ─── Full-screen image preview ───────────────────────────────
        RealBlurPreviewRenderer(
            selectedPhotoUriString = selectedPhotoUriString,
            params = blurParams,
            highQualityDepthEnabled = depthQuality.usesHighQualityPath,
            useInjectedDepth = false,
            displayMode = previewDisplayMode,
            depthRevision = depthRevision,
            activeControlTab = activeControlTab,
            isSaving = isSaving,
            focusPointNormalized = Offset(focusPointX, focusPointY),
            onDepthPicked = { depth, point ->
                focusPointX = point.x
                focusPointY = point.y
                applyBlurParamsChange(blurParams.copy(focusDepth = depth.toFloat()))
            },
            modifier = Modifier.fillMaxSize()
        )

        // ─── Top overlay bar ─────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color.Black.copy(alpha = 0.98f),
                            Color.Black.copy(alpha = 0.82f),
                            Color.Black.copy(alpha = 0.5f),
                            Color.Transparent
                        )
                    )
                )
                .padding(top = statusBarPadding.calculateTopPadding())
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onBack) {
                Icon(
                    Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = Color.White
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            TopActionPill(
                label = if (previewDisplayMode == PreviewDisplayMode.Blur) "Depth" else "Preview",
                active = previewDisplayMode == PreviewDisplayMode.DepthMap,
                onClick = {
                    previewDisplayModeRaw = if (previewDisplayMode == PreviewDisplayMode.Blur)
                        PreviewDisplayMode.DepthMap.name
                    else
                        PreviewDisplayMode.Blur.name
                }
            )

            Spacer(modifier = Modifier.width(8.dp))

            TopActionPill(
                label = "Retouch",
                icon = Icons.Default.Edit,
                onClick = onRetouchDepthClick,
                enabled = selectedPhotoUriString != null && !isSaving,
                disabledAlpha = 0.58f
            )

            Spacer(modifier = Modifier.width(8.dp))

            Box {
                TopActionPill(
                    label = if (isSaving) "Saving" else "Save",
                    icon = Icons.Default.Share,
                    active = true,
                    enabled = selectedPhotoUriString != null && !isSaving,
                    onClick = { saveMenuExpanded = true }
                )

                DropdownMenu(
                    expanded = saveMenuExpanded,
                    onDismissRequest = { saveMenuExpanded = false },
                    modifier = Modifier.background(Color(0xFF242424))
                ) {
                    DropdownMenuItem(
                        text = { Text("Save Standard", color = Color.White) },
                        onClick = {
                            saveMenuExpanded = false
                            selectedPhotoUriString?.let { uriString ->
                                val uri = Uri.parse(uriString)
                                isSaving = true
                                scope.launch {
                                    val cachedDepthOverride =
                                        activeDepthOverrideForQuality()
                                            ?: DepthBlurEngine.cachedPreviewDepth?.takeIf {
                                                DepthBlurEngine.cachedPreviewUriString == uriString
                                                    && !DepthBlurEngine.cachedPreviewUsesInjectedDepth
                                            }
                                    val result = DepthBlurEngine.exportBlurredImage(
                                        context = context,
                                        uri = uri,
                                        params = blurParams,
                                        maxDimension = 1280,
                                        overrideDepth = cachedDepthOverride,
                                        highQualityDepth = depthQuality.usesHighQualityPath
                                    )
                                    isSaving = false
                                    saveStatus = when (result) {
                                        is DepthBlurExportResult.Success -> "Photo saved to gallery."
                                        is DepthBlurExportResult.Error -> "Save failed: ${result.message}"
                                    }
                                    saveDialogVisible = true
                                }
                            }
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Original Resolution", color = AccentAmber) },
                        onClick = {
                            saveMenuExpanded = false
                            selectedPhotoUriString?.let { uriString ->
                                val uri = Uri.parse(uriString)
                                isSaving = true
                                scope.launch {
                                    val cachedDepthOverride =
                                        activeDepthOverrideForQuality()
                                            ?: DepthBlurEngine.cachedPreviewDepth?.takeIf {
                                                DepthBlurEngine.cachedPreviewUriString == uriString
                                                    && !DepthBlurEngine.cachedPreviewUsesInjectedDepth
                                            }
                                    val result = DepthBlurEngine.exportBlurredImage(
                                        context = context,
                                        uri = uri,
                                        params = blurParams,
                                        maxDimension = 2048,
                                        overrideDepth = cachedDepthOverride,
                                        highQualityDepth = depthQuality.usesHighQualityPath
                                    )
                                    isSaving = false
                                    saveStatus = when (result) {
                                        is DepthBlurExportResult.Success -> "High-res photo saved."
                                        is DepthBlurExportResult.Error -> "Save failed: ${result.message}"
                                    }
                                    saveDialogVisible = true
                                }
                            }
                        }
                    )
                }
            }
        }

        // ─── Bottom controls panel (tap handle or image to expand/collapse) ──
        Box(modifier = Modifier.align(Alignment.BottomCenter)) {
            BottomControlsPanel(
                blurParams = blurParams,
                displayMode = previewDisplayMode,
                depthQuality = depthQuality,
                selectedPreset = blurParams.lensEffect.name,
                navBarPadding = navBarPadding.calculateBottomPadding(),
                activeControlTab = activeControlTab,
                onDepthQualityChange = onDepthQualityChange,
                onControlTabChange = {
                    activeControlTabRaw = it.name
                    previewDisplayModeRaw = PreviewDisplayMode.Blur.name
                },
                onPresetChange = { preset ->
                    applyBlurParamsChange(
                        blurParams.copy(lensEffect = preset.effect)
                    )
                },
                onBlurParamsChange = applyBlurParamsChange
            )
        }
    }
}

// ─── Bottom Controls Panel ───────────────────────────────────────────────────

@Composable
private fun BottomControlsPanel(
    blurParams: BlurPreviewParams,
    displayMode: PreviewDisplayMode,
    depthQuality: DepthQualityMode,
    selectedPreset: String,
    navBarPadding: androidx.compose.ui.unit.Dp,
    activeControlTab: PreviewControlTab,
    onDepthQualityChange: (DepthQualityMode) -> Unit,
    onControlTabChange: (PreviewControlTab) -> Unit,
    onPresetChange: (BokehPreset) -> Unit,
    onBlurParamsChange: (BlurPreviewParams) -> Unit,
) {
    val onLensTabClick = remember(onControlTabChange) { { onControlTabChange(PreviewControlTab.Lens) } }
    val onSizeTabClick = remember(onControlTabChange) { { onControlTabChange(PreviewControlTab.Size) } }
    val onEdgeTabClick = remember(onControlTabChange) { { onControlTabChange(PreviewControlTab.Edge) } }
    val onLightTabClick = remember(onControlTabChange) { { onControlTabChange(PreviewControlTab.Light) } }
    val onPresetChangeLambdas = remember(onPresetChange) {
        BokehPreset.entries.associateWith { preset -> { onPresetChange(preset) } }
    }
    val currentBlurParams by androidx.compose.runtime.rememberUpdatedState(blurParams)
    
    val onSizeChange = remember(onBlurParamsChange) {
        { value: Float -> onBlurParamsChange(currentBlurParams.copy(blurStrength = value)) }
    }
    val onTransitionChange = remember(onBlurParamsChange) {
        { value: Float -> onBlurParamsChange(currentBlurParams.copy(blurFalloff = value)) }
    }
    val onSmoothEdgeChange = remember(onBlurParamsChange) {
        { value: Float -> onBlurParamsChange(currentBlurParams.copy(edgeSoftness = value)) }
    }
    val onExpandEdgeChange = remember(onBlurParamsChange) {
        { value: Float -> onBlurParamsChange(currentBlurParams.copy(edgeExpand = value)) }
    }
    val onBgLightChange = remember(onBlurParamsChange) {
        { value: Float -> onBlurParamsChange(currentBlurParams.copy(backgroundLight = value)) }
    }
    val onVignetteChange = remember(onBlurParamsChange) {
        { value: Float -> onBlurParamsChange(currentBlurParams.copy(vignetteStrength = value)) }
    }
    val onFlareChange = remember(onBlurParamsChange) {
        { value: Float -> onBlurParamsChange(currentBlurParams.copy(flareStrength = value)) }
    }
    val onHighlightChange = remember(onBlurParamsChange) {
        { value: Float -> onBlurParamsChange(currentBlurParams.copy(highlightBoost = value)) }
    }

    Surface(
        modifier = Modifier
            .fillMaxWidth(),
        shape = RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp),
        color = Color(0xD2141414),
        tonalElevation = 0.dp
    ) {
        Column(
            modifier = Modifier
                .padding(horizontal = 14.dp, vertical = 12.dp)
                .padding(bottom = navBarPadding + 2.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (displayMode == PreviewDisplayMode.DepthMap) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(18.dp))
                        .background(Color.White.copy(alpha = 0.06f))
                        .padding(horizontal = 14.dp, vertical = 10.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Column {
                        Text(
                            text = "Depth quality",
                            color = Color.White,
                            style = MaterialTheme.typography.titleSmall
                        )
                        Text(
                            text = depthQuality.description,
                            color = Color.White.copy(alpha = 0.65f),
                            style = MaterialTheme.typography.bodySmall
                        )
                    }

                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        val context = LocalContext.current
                        FilterChip(
                            selected = depthQuality == DepthQualityMode.Standard,
                            onClick = { onDepthQualityChange(DepthQualityMode.Standard) },
                            label = { Text(DepthQualityMode.Standard.label) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = AccentAmber,
                                selectedLabelColor = Color.Black,
                                containerColor = Color.White.copy(alpha = 0.06f),
                                labelColor = Color.White.copy(alpha = 0.72f)
                            )
                        )
                        FilterChip(
                            selected = depthQuality == DepthQualityMode.Hd,
                            onClick = { onDepthQualityChange(DepthQualityMode.Hd) },
                            label = { Text(DepthQualityMode.Hd.label) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = AccentAmber,
                                selectedLabelColor = Color.Black,
                                containerColor = Color.White.copy(alpha = 0.06f),
                                labelColor = Color.White.copy(alpha = 0.72f)
                            )
                        )
                        FilterChip(
                            selected = false,
                            onClick = {
                                Toast.makeText(context, "Ultra (Depth Pro) — coming soon", Toast.LENGTH_SHORT).show()
                            },
                            label = { Text(DepthQualityMode.Ultra.label) },
                            colors = FilterChipDefaults.filterChipColors(
                                containerColor = Color.White.copy(alpha = 0.06f),
                                labelColor = AccentAmber.copy(alpha = 0.72f)
                            )
                        )
                    }
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                ControlTabChip(
                    label = "Lens",
                    selected = activeControlTab == PreviewControlTab.Lens,
                    onClick = onLensTabClick,
                    modifier = Modifier.weight(1f)
                )
                ControlTabChip(
                    label = "Size",
                    selected = activeControlTab == PreviewControlTab.Size,
                    onClick = onSizeTabClick,
                    modifier = Modifier.weight(1f)
                )
                ControlTabChip(
                    label = "Edge",
                    selected = activeControlTab == PreviewControlTab.Edge,
                    onClick = onEdgeTabClick,
                    modifier = Modifier.weight(1f)
                )
                ControlTabChip(
                    label = "Light",
                    selected = activeControlTab == PreviewControlTab.Light,
                    onClick = onLightTabClick,
                    modifier = Modifier.weight(1f)
                )
            }

            when (activeControlTab) {
                PreviewControlTab.Lens -> {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        // Master Lens Profiles
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState()),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            LensProfile.entries.forEach { profile ->
                                val profileClick = remember(onBlurParamsChange, onPresetChange, profile) {
                                    {
                                        val p = profile.params
                                        onBlurParamsChange(p.copy(focusDepth = currentBlurParams.focusDepth))
                                        val matchPreset = BokehPreset.entries.firstOrNull { it.effect == p.lensEffect }
                                        if (matchPreset != null) onPresetChange(matchPreset)
                                    }
                                }
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(Color.White.copy(alpha = 0.06f))
                                        .border(
                                            1.dp,
                                            Color.White.copy(alpha = 0.10f),
                                            RoundedCornerShape(12.dp)
                                        )
                                        .clickable(onClick = profileClick)
                                        .padding(horizontal = 14.dp, vertical = 8.dp)
                                ) {
                                    Text(
                                        text = profile.label,
                                        style = MaterialTheme.typography.labelMedium,
                                        color = AccentAmber.copy(alpha = 0.85f),
                                        fontWeight = FontWeight.Medium
                                    )
                                }
                            }
                        }
                        // Bokeh Shape Cards
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState()),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            BokehPreset.entries.forEach { preset ->
                                val isSelected = selectedPreset == preset.effect.name
                                LensEffectCard(
                                    preset = preset,
                                    selected = isSelected,
                                    onClick = onPresetChangeLambdas[preset]!!
                                )
                            }
                        }
                    }
                }

                PreviewControlTab.Size -> {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        OverlaySliderRow(
                            label = "Size",
                            value = blurParams.blurStrength,
                            valueLabel = "${(blurParams.blurStrength * 100).toInt()}%",
                            onValueChange = onSizeChange
                        )
                        OverlaySliderRow(
                            label = "Transition",
                            value = blurParams.blurFalloff,
                            valueLabel = "${(blurParams.blurFalloff * 100).toInt()}%",
                            onValueChange = onTransitionChange
                        )
                    }
                }

                PreviewControlTab.Edge -> {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            OverlaySliderRow(
                                label = "Smooth edge",
                                value = blurParams.edgeSoftness,
                                valueLabel = "${(blurParams.edgeSoftness * 100).toInt()}%",
                                onValueChange = onSmoothEdgeChange,
                                modifier = Modifier.weight(1f)
                            )
                            OverlaySliderRow(
                                label = "Expand edge",
                                value = blurParams.edgeExpand,
                                valueLabel = "${(blurParams.edgeExpand * 100).toInt()}%",
                                onValueChange = onExpandEdgeChange,
                                modifier = Modifier.weight(1f)
                            )
                        }
                        OverlaySliderRow(
                            label = "Smart refine",
                            value = blurParams.edgeRefine,
                            valueLabel = "${(blurParams.edgeRefine * 100).toInt()}%",
                            onValueChange = { value ->
                                onBlurParamsChange(blurParams.copy(edgeRefine = value))
                            }
                        )
                    }
                }

                PreviewControlTab.Light -> {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        OverlaySliderRow(
                            label = "Background light",
                            value = blurParams.backgroundLight,
                            valueLabel = "${(blurParams.backgroundLight * 100).toInt()}%",
                            onValueChange = onBgLightChange
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            OverlaySliderRow(
                                label = "Vignette",
                                value = blurParams.vignetteStrength,
                                valueLabel = "${(blurParams.vignetteStrength * 100).toInt()}%",
                                onValueChange = onVignetteChange,
                                modifier = Modifier.weight(1f)
                            )
                            OverlaySliderRow(
                                label = "Boost highlights",
                                value = blurParams.highlightBoost,
                                valueLabel = "${(blurParams.highlightBoost * 100).toInt()}%",
                                onValueChange = onHighlightChange,
                                modifier = Modifier.weight(1f)
                            )
                        }
                        OverlaySliderRow(
                            label = "Lens flare",
                            value = blurParams.flareStrength,
                            valueLabel = "${(blurParams.flareStrength * 100).toInt()}%",
                            onValueChange = onFlareChange
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun TopActionPill(
    label: String,
    icon: ImageVector? = null,
    active: Boolean = false,
    enabled: Boolean = true,
    disabledAlpha: Float = 0.38f,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        shape = RoundedCornerShape(999.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (active) AccentAmber else Color.White.copy(alpha = 0.16f),
            contentColor = if (active) Color(0xFF1A1200) else Color.White,
            disabledContainerColor = if (active) AccentAmber.copy(alpha = disabledAlpha) else Color.White.copy(alpha = 0.12f),
            disabledContentColor = Color.White.copy(alpha = disabledAlpha)
        ),
        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 0.dp)
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(modifier = Modifier.width(6.dp))
        }
        Text(
            label,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
private fun ControlTabChip(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(
                if (selected) AccentAmber.copy(alpha = 0.15f)
                else Color.White.copy(alpha = 0.06f)
            )
            .border(
                width = 1.dp,
                color = if (selected) AccentAmber.copy(alpha = 0.4f) else Color.Transparent,
                shape = RoundedCornerShape(14.dp)
            )
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = if (selected) AccentAmber else Color.White.copy(alpha = 0.7f),
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal
        )
    }
}

@Composable
private fun LensEffectCard(
    preset: BokehPreset,
    selected: Boolean,
    onClick: () -> Unit
) {
    val cardColor = if (selected) {
        AccentAmber.copy(alpha = 0.16f)
    } else {
        Color.White.copy(alpha = 0.06f)
    }
    val strokeColor = if (selected) AccentAmber else Color.White.copy(alpha = 0.12f)

    Column(
        modifier = Modifier
            .width(78.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(cardColor)
            .border(1.dp, strokeColor, RoundedCornerShape(16.dp))
            .clickable(onClick = onClick)
            .padding(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Canvas(
            modifier = Modifier
                .size(60.dp)
                .clip(RoundedCornerShape(12.dp))
        ) {
            drawRect(
                brush = Brush.linearGradient(
                    colors = listOf(Color(0xFF2B3440), Color(0xFF161A1F)),
                    start = Offset.Zero,
                    end = Offset(size.width, size.height)
                )
            )
            drawCircle(
                color = Color(0xFFF3CC7A).copy(alpha = 0.9f),
                radius = size.minDimension * 0.12f,
                center = Offset(size.width * 0.26f, size.height * 0.28f)
            )
            when (preset.effect) {
                LensEffect.Classic -> {
                    repeat(4) { index ->
                        drawCircle(
                            color = Color.White.copy(alpha = 0.24f + index * 0.08f),
                            radius = size.minDimension * (0.12f + index * 0.03f),
                            center = Offset(size.width * 0.66f, size.height * 0.62f)
                        )
                    }
                }
                LensEffect.Creamy -> {
                    repeat(5) { index ->
                        drawCircle(
                            color = Color(0xFFFFF0CF).copy(alpha = 0.16f + index * 0.05f),
                            radius = size.minDimension * (0.15f + index * 0.035f),
                            center = Offset(size.width * 0.62f, size.height * 0.58f)
                        )
                    }
                }
                LensEffect.Bubble -> {
                    repeat(5) { index ->
                        val radius = size.minDimension * (0.10f + index * 0.02f)
                        drawCircle(
                            color = Color(0xFFFFE29A).copy(alpha = 0.22f),
                            radius = radius,
                            center = Offset(size.width * 0.62f, size.height * 0.58f),
                            style = Stroke(width = 1.5.dp.toPx())
                        )
                    }
                }
                LensEffect.Bloom -> {
                    drawCircle(
                        brush = Brush.radialGradient(
                            colors = listOf(
                                Color(0xFFFFF5D6).copy(alpha = 0.86f),
                                Color(0xFFFFD46D).copy(alpha = 0.18f),
                                Color.Transparent
                            ),
                            center = Offset(size.width * 0.62f, size.height * 0.56f),
                            radius = size.minDimension * 0.42f
                        ),
                        radius = size.minDimension * 0.42f,
                        center = Offset(size.width * 0.62f, size.height * 0.56f)
                    )
                }
                LensEffect.Star -> {
                    val center = Offset(size.width * 0.62f, size.height * 0.58f)
                    drawLine(
                        color = Color.White.copy(alpha = 0.34f),
                        start = Offset(center.x - 15.dp.toPx(), center.y),
                        end = Offset(center.x + 15.dp.toPx(), center.y),
                        strokeWidth = 2.dp.toPx()
                    )
                    drawLine(
                        color = Color.White.copy(alpha = 0.34f),
                        start = Offset(center.x, center.y - 15.dp.toPx()),
                        end = Offset(center.x, center.y + 15.dp.toPx()),
                        strokeWidth = 2.dp.toPx()
                    )
                    drawCircle(
                        color = Color(0xFFFFE29A).copy(alpha = 0.34f),
                        radius = size.minDimension * 0.11f,
                        center = center
                    )
                }
                LensEffect.Hexagon -> {
                    val center = Offset(size.width * 0.62f, size.height * 0.58f)
                    repeat(3) { index ->
                        val radius = size.minDimension * (0.18f + index * 0.08f)
                        drawCircle(
                            color = Color(0xFFF3CC7A).copy(alpha = 0.2f + index * 0.1f),
                            radius = radius,
                            center = center,
                            style = Stroke(width = 1.5.dp.toPx())
                        )
                    }
                }
                LensEffect.Anamorphic -> {
                    val center = Offset(size.width * 0.62f, size.height * 0.58f)
                    // Oval bokeh
                    drawOval(
                        color = Color(0xFFF3CC7A).copy(alpha = 0.32f),
                        topLeft = Offset(center.x - 16.dp.toPx(), center.y - 8.dp.toPx()),
                        size = Size(32.dp.toPx(), 16.dp.toPx()),
                        style = Stroke(width = 1.5.dp.toPx())
                    )
                    // Horizontal flare streak
                    drawLine(
                        brush = Brush.horizontalGradient(
                            colors = listOf(
                                Color.Transparent,
                                Color(0xFF8EC5FC).copy(alpha = 0.45f),
                                Color(0xFFE0C3FC).copy(alpha = 0.50f),
                                Color(0xFF8EC5FC).copy(alpha = 0.45f),
                                Color.Transparent
                            ),
                            startX = 0f,
                            endX = size.width
                        ),
                        start = Offset(0f, center.y),
                        end = Offset(size.width, center.y),
                        strokeWidth = 3.dp.toPx()
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = preset.label,
            style = MaterialTheme.typography.labelSmall,
            color = if (selected) AccentAmber else Color.White.copy(alpha = 0.74f),
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal
        )
    }
}

// ─── Display Mode Chip ───────────────────────────────────────────────────────

@Composable
private fun DisplayModeChip(
    icon: ImageVector,
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(
                if (selected) AccentAmber.copy(alpha = 0.15f)
                else Color.White.copy(alpha = 0.05f)
            )
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            icon,
            contentDescription = label,
            tint = if (selected) AccentAmber else Color.White.copy(alpha = 0.5f),
            modifier = Modifier.size(22.dp)
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = if (selected) AccentAmber else Color.White.copy(alpha = 0.5f),
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal
        )
    }
}

// ─── Overlay Slider Row ──────────────────────────────────────────────────────

@Composable
private fun OverlaySliderRow(
    label: String,
    value: Float,
    valueLabel: String,
    onValueChange: (Float) -> Unit,
    range: ClosedFloatingPointRange<Float> = 0f..1f,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier.padding(vertical = 1.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                label,
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.7f)
            )
            Text(
                valueLabel,
                style = MaterialTheme.typography.labelMedium,
                color = AccentAmber,
                fontWeight = FontWeight.SemiBold
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = range,
            modifier = Modifier.height(32.dp),
            colors = SliderDefaults.colors(
                thumbColor = AccentAmber,
                activeTrackColor = AccentAmber,
                inactiveTrackColor = Color.White.copy(alpha = 0.1f)
            )
        )
    }
}

// ─── Depth Map Thumbnail ─────────────────────────────────────────────────────

@Composable
private fun DepthMapThumbnail(
    selectedPhotoUriString: String?,
    params: BlurPreviewParams,
    depthRevision: Int,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val photoUri = selectedPhotoUriString?.let(Uri::parse)
    var depthBitmap by remember(selectedPhotoUriString) { mutableStateOf<Bitmap?>(null) }

    DisposableEffect(selectedPhotoUriString) {
        onDispose { depthBitmap = null }
    }

    LaunchedEffect(photoUri, depthRevision) {
        if (photoUri == null) {
            depthBitmap = null; return@LaunchedEffect
        }
        try {
            val result = DepthBlurEngine.renderPreview(
                context = context,
                uri = photoUri,
                params = params,
                maxDimension = 200,
                overrideDepth = DepthBlurEngine.getModifiedDepth(),
                updateCache = false
            )
            depthBitmap = result.depthMapBitmap
        } catch (_: Exception) {
        }
    }

    depthBitmap?.let { db ->
        Box(
            modifier = modifier
                .clip(RoundedCornerShape(12.dp))
                .background(GlassBlack)
                .padding(4.dp)
        ) {
            Image(
                bitmap = db.asImageBitmap(),
                contentDescription = "Depth map",
                contentScale = ContentScale.Crop,
                modifier = Modifier
                    .width(80.dp)
                    .height(54.dp)
                    .clip(RoundedCornerShape(8.dp))
            )
        }
    }
}

// ─── Full-Screen Blur Preview Renderer ───────────────────────────────────────

@Composable
private fun RealBlurPreviewRenderer(
    selectedPhotoUriString: String?,
    params: BlurPreviewParams,
    highQualityDepthEnabled: Boolean,
    useInjectedDepth: Boolean,
    displayMode: PreviewDisplayMode,
    depthRevision: Int,
    activeControlTab: PreviewControlTab,
    isSaving: Boolean = false,
    focusPointNormalized: Offset = Offset(0.5f, 0.42f),
    onDepthPicked: (Int, Offset) -> Unit = { _, _ -> },
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val photoUri = selectedPhotoUriString?.let(Uri::parse)
    var baseSourceBitmap by remember(selectedPhotoUriString, highQualityDepthEnabled, useInjectedDepth) {
        mutableStateOf(cachedPreviewCopy(selectedPhotoUriString, highQualityDepthEnabled, useInjectedDepth, DepthBlurEngine.cachedPreviewSource))
    }
    var renderedBitmap by remember(selectedPhotoUriString, highQualityDepthEnabled, useInjectedDepth) {
        mutableStateOf(cachedPreviewCopy(selectedPhotoUriString, highQualityDepthEnabled, useInjectedDepth, DepthBlurEngine.cachedPreviewRendered))
    }
    var baseDepthBitmap by remember(selectedPhotoUriString, highQualityDepthEnabled, useInjectedDepth) {
        mutableStateOf(cachedPreviewCopy(selectedPhotoUriString, highQualityDepthEnabled, useInjectedDepth, DepthBlurEngine.cachedPreviewDepth))
    }
    var renderError by remember(selectedPhotoUriString) { mutableStateOf<String?>(null) }
    var isRendering by remember(selectedPhotoUriString) { mutableStateOf(false) }
    var edgeGuideBitmap by remember(selectedPhotoUriString) { mutableStateOf<Bitmap?>(null) }
    var focusGuideBitmap by remember(selectedPhotoUriString) { mutableStateOf<Bitmap?>(null) }
    var lastRenderedParams by remember(selectedPhotoUriString, depthRevision) {
        mutableStateOf<BlurPreviewParams?>(null)
    }

    // Focus ring animation state
    var focusTapPos by remember { mutableStateOf<Offset?>(null) }
    val focusRingScale = remember { Animatable(0f) }
    val focusRingAlpha = remember { Animatable(0f) }
    val focusIndicatorAlpha = remember { Animatable(0f) }
    val scope = rememberCoroutineScope()
    var isCompareMode by remember { mutableStateOf(false) }
    var compareSliderFraction by remember { mutableFloatStateOf(0.5f) }
    var containerSize by remember { mutableStateOf(IntSize.Zero) }

    DisposableEffect(selectedPhotoUriString) {
        onDispose {
            baseSourceBitmap?.recycle()
            renderedBitmap?.recycle()
            baseDepthBitmap?.recycle()
            edgeGuideBitmap?.recycle()
            focusGuideBitmap?.recycle()
            baseSourceBitmap = null
            renderedBitmap = null
            baseDepthBitmap = null
            edgeGuideBitmap = null
            focusGuideBitmap = null
        }
    }

    // ─── Immediate source decode for instant UI feedback ───
    LaunchedEffect(photoUri) {
        if (photoUri == null) return@LaunchedEffect
        // Only decode if we don't have it in cache yet
        if (baseSourceBitmap == null) {
            try {
                // We use a small dimension first or full screen to show it instantly
                val decoded = withContext(Dispatchers.IO) {
                    DepthBlurEngine.decodeScaledBitmap(context, photoUri, 960)
                }
                baseSourceBitmap = baseSourceBitmap.replaceWith(decoded)
            } catch (_: Exception) {}
        }
    }

    LaunchedEffect(photoUri, depthRevision, highQualityDepthEnabled, useInjectedDepth) {
        if (photoUri == null) {
            baseSourceBitmap = null
            renderedBitmap = null
            baseDepthBitmap = null
            lastRenderedParams = null
            renderError = null; isRendering = false
            return@LaunchedEffect
        }
        isRendering = true
        renderError = null
        try {
            val result = DepthBlurEngine.renderPreview(
                context = context,
                uri = photoUri,
                params = params,
                maxDimension = if (highQualityDepthEnabled) 960 else 480,
                overrideDepth = if (useInjectedDepth) {
                    DepthBlurEngine.getPreferredDepthOverride()
                } else {
                    DepthBlurEngine.getModifiedDepth()
                },
                highQualityDepth = highQualityDepthEnabled,
                usesInjectedDepth = useInjectedDepth
            )
            // If we already decoded the source, replace it with the one used in render
            baseSourceBitmap = baseSourceBitmap.replaceWith(result.sourceBitmap)
            renderedBitmap = renderedBitmap.replaceWith(result.bitmap)
            baseDepthBitmap = baseDepthBitmap.replaceWith(result.depthMapBitmap)
            lastRenderedParams = params
        } catch (c: kotlinx.coroutines.CancellationException) {
            throw c
        } catch (t: Throwable) {
            renderError = t.message ?: "Preview failed"
        } finally {
            isRendering = false
        }
    }

    LaunchedEffect(params, baseSourceBitmap, baseDepthBitmap) {
        val src = baseSourceBitmap ?: return@LaunchedEffect
        val depth = baseDepthBitmap ?: return@LaunchedEffect
        if (lastRenderedParams == params && renderedBitmap != null) return@LaunchedEffect

        if (renderedBitmap != null) {
            delay(120)
        }
        isRendering = true
        renderError = null
        try {
            val rerenderedBitmap = DepthBlurEngine.rerenderPreviewFromCache(
                sourceBitmap = src,
                depthBitmap = depth,
                params = params
            )
            renderedBitmap = renderedBitmap.replaceWith(rerenderedBitmap)
            lastRenderedParams = params
        } catch (c: kotlinx.coroutines.CancellationException) {
            throw c
        } catch (t: Throwable) {
            renderError = t.message ?: "Preview failed"
        } finally {
            isRendering = false
        }
    }

    LaunchedEffect(activeControlTab, baseDepthBitmap, params.focusDepth, params.edgeSoftness, params.edgeExpand) {
        val depthBitmap = baseDepthBitmap
        if (activeControlTab != PreviewControlTab.Edge || depthBitmap == null) {
            edgeGuideBitmap?.recycle()
            edgeGuideBitmap = null
            return@LaunchedEffect
        }
        val guide = withContext(Dispatchers.Default) {
            createEdgeGuideBitmap(depthBitmap, params)
        }
        edgeGuideBitmap = edgeGuideBitmap.replaceWith(guide)
    }

    LaunchedEffect(activeControlTab, baseDepthBitmap, params.focusDepth, params.blurFalloff) {
        val depthBitmap = baseDepthBitmap
        if (activeControlTab != PreviewControlTab.Size || depthBitmap == null) {
            focusGuideBitmap?.recycle()
            focusGuideBitmap = null
            return@LaunchedEffect
        }
        val guide = withContext(Dispatchers.Default) {
            createFocusGuideBitmap(depthBitmap, params)
        }
        focusGuideBitmap = focusGuideBitmap.replaceWith(guide)
    }

    Box(
        modifier = modifier
            .onSizeChanged { containerSize = it }
            .pointerInput(baseDepthBitmap, displayMode) {
                detectTapGestures(
                    onTap = { offset ->
                        if (displayMode != PreviewDisplayMode.Blur) return@detectTapGestures
                        baseDepthBitmap?.let { db ->
                            val imageRect = computeFittedImageRect(
                                IntSize(size.width.toInt(), size.height.toInt()),
                                db
                            ) ?: return@let
                            if (!imageRect.contains(offset)) return@let
                            val normalizedPoint = Offset(
                                x = ((offset.x - imageRect.left) / imageRect.width).coerceIn(0f, 1f),
                                y = ((offset.y - imageRect.top) / imageRect.height).coerceIn(0f, 1f)
                            )
                            val x = (normalizedPoint.x * (db.width - 1)).toInt().coerceIn(0, db.width - 1)
                            val y = (normalizedPoint.y * (db.height - 1)).toInt().coerceIn(0, db.height - 1)
                            val pixel = db.getPixel(x, y)
                            val depth = android.graphics.Color.red(pixel)
                            onDepthPicked(depth, normalizedPoint)

                            focusTapPos = Offset(
                                imageRect.left + normalizedPoint.x * imageRect.width,
                                imageRect.top + normalizedPoint.y * imageRect.height
                            )
                            scope.launch {
                                focusRingScale.snapTo(0.3f)
                                focusRingAlpha.snapTo(1f)
                                focusIndicatorAlpha.snapTo(1f)
                                focusRingScale.animateTo(
                                    1f,
                                    animationSpec = tween(350, easing = FastOutSlowInEasing)
                                )
                            }
                            scope.launch {
                                delay(1000)
                                focusRingAlpha.animateTo(
                                    0f,
                                    animationSpec = tween(300)
                                )
                                focusIndicatorAlpha.animateTo(
                                    0.15f,
                                    animationSpec = tween(500)
                                )
                                focusTapPos = null
                            }

                        }
                    }
                )
            },
        contentAlignment = Alignment.Center
    ) {
        when {
            displayMode == PreviewDisplayMode.DepthMap && baseDepthBitmap != null -> {
                Image(
                    bitmap = baseDepthBitmap!!.asImageBitmap(),
                    contentDescription = "Depth map",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize()
                )
            }

            isCompareMode && renderedBitmap != null && baseSourceBitmap != null -> {
                // Before/After comparison slider
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .pointerInput(Unit) {
                            detectDragGestures { change, _ ->
                                change.consume()
                                compareSliderFraction =
                                    (change.position.x / size.width).coerceIn(0f, 1f)
                            }
                        }
                ) {
                    // Rendered (After) — full background
                    Image(
                        bitmap = renderedBitmap!!.asImageBitmap(),
                        contentDescription = "Blurred preview",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.fillMaxSize()
                    )
                    // Original (Before) — clipped to left of slider
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .drawWithContent {
                                val sliderX = size.width * compareSliderFraction
                                clipRect(right = sliderX) {
                                    this@drawWithContent.drawContent()
                                }
                            }
                    ) {
                        Image(
                            bitmap = baseSourceBitmap!!.asImageBitmap(),
                            contentDescription = "Original",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                    // Slider divider line + handle
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val sliderX = size.width * compareSliderFraction
                        drawLine(
                            color = Color.White,
                            start = Offset(sliderX, 0f),
                            end = Offset(sliderX, size.height),
                            strokeWidth = 2.dp.toPx()
                        )
                        // Handle circle
                        drawCircle(
                            color = Color.White,
                            radius = 14.dp.toPx(),
                            center = Offset(sliderX, size.height / 2f)
                        )
                        drawCircle(
                            color = Color.Black.copy(alpha = 0.4f),
                            radius = 12.dp.toPx(),
                            center = Offset(sliderX, size.height / 2f)
                        )
                        // Arrows
                        val arrowY = size.height / 2f
                        val arrowSize = 5.dp.toPx()
                        // Left arrow
                        drawLine(Color.White, Offset(sliderX - 7.dp.toPx(), arrowY), Offset(sliderX - 2.dp.toPx(), arrowY - arrowSize), 1.5.dp.toPx())
                        drawLine(Color.White, Offset(sliderX - 7.dp.toPx(), arrowY), Offset(sliderX - 2.dp.toPx(), arrowY + arrowSize), 1.5.dp.toPx())
                        // Right arrow
                        drawLine(Color.White, Offset(sliderX + 7.dp.toPx(), arrowY), Offset(sliderX + 2.dp.toPx(), arrowY - arrowSize), 1.5.dp.toPx())
                        drawLine(Color.White, Offset(sliderX + 7.dp.toPx(), arrowY), Offset(sliderX + 2.dp.toPx(), arrowY + arrowSize), 1.5.dp.toPx())
                    }
                    // Labels
                    Text(
                        "Before",
                        color = Color.White.copy(alpha = 0.8f),
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(start = 12.dp, top = 8.dp)
                            .background(Color.Black.copy(alpha = 0.4f), RoundedCornerShape(6.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                    Text(
                        "After",
                        color = Color.White.copy(alpha = 0.8f),
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(end = 12.dp, top = 8.dp)
                            .background(Color.Black.copy(alpha = 0.4f), RoundedCornerShape(6.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            renderedBitmap != null -> {
                Image(
                    bitmap = renderedBitmap!!.asImageBitmap(),
                    contentDescription = "Blurred preview",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize()
                )
            }

            photoUri != null -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color(0xFF0A0A0A)),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(40.dp),
                        strokeWidth = 3.dp,
                        color = AccentAmber
                    )
                }
            }

            else -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color(0xFF0A0A0A))
                )
            }
        }

        if (displayMode == PreviewDisplayMode.Blur && baseSourceBitmap != null && containerSize != IntSize.Zero) {
            val imageRect = computeFittedImageRect(containerSize, baseSourceBitmap!!)
            val focusCenter = imageRect?.let { rect ->
                Offset(
                    x = rect.left + rect.width * focusPointNormalized.x.coerceIn(0f, 1f),
                    y = rect.top + rect.height * focusPointNormalized.y.coerceIn(0f, 1f)
                )
            }
            if (focusCenter != null) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val ringRadius = 28.dp.toPx() + 18.dp.toPx() * params.blurFalloff
                    val indicatorAlpha = focusIndicatorAlpha.value
                    
                    if (indicatorAlpha > 0f) {
                        drawCircle(
                            color = Color(0xFF5CFF98).copy(alpha = 0.9f * indicatorAlpha),
                            radius = ringRadius,
                            center = focusCenter,
                            style = Stroke(width = 2.dp.toPx())
                        )
                        drawCircle(
                            color = Color(0xFF5CFF98).copy(alpha = 0.28f * indicatorAlpha),
                            radius = ringRadius * 0.58f,
                            center = focusCenter,
                            style = Stroke(width = 1.dp.toPx())
                        )
                        drawCircle(
                            color = Color(0xFFFF4B4B).copy(alpha = indicatorAlpha),
                            radius = 5.dp.toPx(),
                            center = focusCenter
                        )
                        drawCircle(
                            color = Color.White.copy(alpha = 0.9f * indicatorAlpha),
                            radius = 1.5.dp.toPx(),
                            center = focusCenter
                        )
                    }
                }
            }
        }

        if (
            displayMode == PreviewDisplayMode.Blur &&
            activeControlTab == PreviewControlTab.Edge &&
            edgeGuideBitmap != null
        ) {
            Image(
                bitmap = edgeGuideBitmap!!.asImageBitmap(),
                contentDescription = "Edge guide",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize()
            )
        }

        if (
            displayMode == PreviewDisplayMode.Blur &&
            activeControlTab == PreviewControlTab.Size &&
            focusGuideBitmap != null
        ) {
            Image(
                bitmap = focusGuideBitmap!!.asImageBitmap(),
                contentDescription = "Focus guide",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize()
            )
        }

        // Focus ring overlay
        focusTapPos?.let { pos ->
            Canvas(modifier = Modifier.fillMaxSize()) {
                val ringRadius = 32.dp.toPx() * focusRingScale.value
                val alpha = focusRingAlpha.value
                drawCircle(
                    color = AccentAmber.copy(alpha = alpha * 0.9f),
                    radius = ringRadius,
                    center = pos,
                    style = Stroke(width = 2.dp.toPx())
                )
                drawCircle(
                    color = AccentAmber.copy(alpha = alpha * 0.3f),
                    radius = ringRadius * 0.55f,
                    center = pos,
                    style = Stroke(width = 1.dp.toPx())
                )
                // Crosshair
                val crossLen = 7.dp.toPx()
                drawLine(
                    AccentAmber.copy(alpha = alpha * 0.7f),
                    start = Offset(pos.x - crossLen, pos.y),
                    end = Offset(pos.x + crossLen, pos.y),
                    strokeWidth = 1.dp.toPx()
                )
                drawLine(
                    AccentAmber.copy(alpha = alpha * 0.7f),
                    start = Offset(pos.x, pos.y - crossLen),
                    end = Offset(pos.x, pos.y + crossLen),
                    strokeWidth = 1.dp.toPx()
                )
            }
        }

        // Loading overlays
        if (isSaving) {
            SavingLoadingEffect()
        } else if (isRendering && renderedBitmap == null) {
            if (baseSourceBitmap != null) {
                // Show original image while rendering for better UX
                Image(
                    bitmap = baseSourceBitmap!!.asImageBitmap(),
                    contentDescription = "Source",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize()
                )
                // Add anime/scan effect
                AnimeLoadingEffect()
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.4f)),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(36.dp),
                        strokeWidth = 3.dp,
                        color = AccentAmber
                    )
                }
            }
        }

        // Error overlay
        if (renderError != null) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color(0xCC331111))
                    .padding(horizontal = 16.dp, vertical = 10.dp)
            ) {
                Text(
                    text = renderError!!,
                    color = Color(0xFFFF6B6B),
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }

        if (displayMode == PreviewDisplayMode.Blur && baseSourceBitmap != null) {
            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(top = 96.dp, end = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Compare button
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(999.dp))
                        .background(
                            if (isCompareMode) AccentAmber.copy(alpha = 0.22f)
                            else Color.Black.copy(alpha = 0.48f)
                        )
                        .clickable {
                            isCompareMode = !isCompareMode
                            compareSliderFraction = 0.5f
                        }
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(if (isCompareMode) AccentAmber else Color.White.copy(alpha = 0.35f))
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Compare",
                            color = Color.White.copy(alpha = 0.9f),
                            style = MaterialTheme.typography.labelMedium
                        )
                    }
                }
            }
        }

        // Benchmark overlay (bottom-left, tap to toggle)
        var showBenchmark by remember { mutableStateOf(false) }
        if (displayMode == PreviewDisplayMode.Blur && renderedBitmap != null) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(start = 12.dp, bottom = 220.dp)
            ) {
                if (showBenchmark) {
                    val runtime = Runtime.getRuntime()
                    val usedMb = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
                    Column(
                        modifier = Modifier
                            .clip(RoundedCornerShape(10.dp))
                            .background(Color.Black.copy(alpha = 0.72f))
                            .clickable { showBenchmark = false }
                            .padding(horizontal = 10.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        Text(
                            "Depth  ${DepthBlurEngine.lastDepthInferenceMs} ms",
                            color = Color(0xFF5CFF98),
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            "Shader ${DepthBlurEngine.lastShaderPassMs} ms",
                            color = Color(0xFF5CFF98),
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            "Total  ${DepthBlurEngine.lastTotalRenderMs} ms",
                            color = AccentAmber,
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            "Mem    ${usedMb} MB",
                            color = Color.White.copy(alpha = 0.7f),
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                } else {
                    Text(
                        "Perf",
                        color = Color.White.copy(alpha = 0.5f),
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(Color.Black.copy(alpha = 0.4f))
                            .clickable { showBenchmark = true }
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }
        }
    }
}

private fun activeDepthOverrideForQuality(): Bitmap? {
    return DepthBlurEngine.getModifiedDepth()
}

private fun cachedPreviewCopy(
    selectedPhotoUriString: String?,
    highQualityDepthEnabled: Boolean,
    usesInjectedDepth: Boolean,
    bitmap: Bitmap?
): Bitmap? {
    if (
        selectedPhotoUriString == null ||
        DepthBlurEngine.cachedPreviewUriString != selectedPhotoUriString ||
        DepthBlurEngine.cachedPreviewHighQualityDepth != highQualityDepthEnabled ||
        DepthBlurEngine.cachedPreviewUsesInjectedDepth != usesInjectedDepth
    ) {
        return null
    }
    if (bitmap == null || bitmap.isRecycled) return null
    return bitmap.copy(Bitmap.Config.ARGB_8888, false)
}

private fun computeFittedImageRect(containerSize: IntSize, bitmap: Bitmap): androidx.compose.ui.geometry.Rect? {
    if (
        containerSize.width <= 0 ||
        containerSize.height <= 0 ||
        bitmap.width <= 0 ||
        bitmap.height <= 0
    ) {
        return null
    }
    val scale = min(
        containerSize.width.toFloat() / bitmap.width.toFloat(),
        containerSize.height.toFloat() / bitmap.height.toFloat()
    )
    val drawWidth = bitmap.width * scale
    val drawHeight = bitmap.height * scale
    val left = (containerSize.width - drawWidth) / 2f
    val top = (containerSize.height - drawHeight) / 2f
    return androidx.compose.ui.geometry.Rect(
        left = left,
        top = top,
        right = left + drawWidth,
        bottom = top + drawHeight
    )
}

private fun createEdgeGuideBitmap(depthBitmap: Bitmap, params: BlurPreviewParams): Bitmap {
    val workBitmap = scaleGuideSource(depthBitmap, 640)
    val width = workBitmap.width
    val height = workBitmap.height
    val pixels = IntArray(width * height)
    workBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
    val focus = params.focusDepth.coerceIn(0f, 255f) / 255f
    val transitionBand = params.blurFalloff.coerceIn(0f, 1f) * 0.050f
    val smoothBand = 0.018f + params.edgeSoftness.coerceIn(0f, 1f) * 0.065f + transitionBand
    val expandBand = params.edgeExpand.coerceIn(0f, 1f) * 0.085f
    val refineBias = params.edgeRefine.coerceIn(0f, 1f) * 0.018f
    val innerBand = smoothBand + expandBand
    val outerBand = innerBand + 0.045f + refineBias
    val out = IntArray(width * height)

    fun isInside(index: Int, band: Float): Boolean {
        val d = ((pixels[index] shr 16) and 0xFF) / 255f
        return kotlin.math.abs(d - focus) <= band
    }

    for (y in 0 until height) {
        for (x in 0 until width) {
            val index = y * width + x
            val inside = isInside(index, innerBand)
            val near = isInside(index, outerBand)
            if (!inside && !near) continue

            var isBoundary = false
            if (inside) {
                val left = x > 0 && !isInside(index - 1, innerBand)
                val right = x < width - 1 && !isInside(index + 1, innerBand)
                val top = y > 0 && !isInside(index - width, innerBand)
                val bottom = y < height - 1 && !isInside(index + width, innerBand)
                isBoundary = left || right || top || bottom
            }

            out[index] = when {
                isBoundary -> android.graphics.Color.argb(255, 255, 72, 72)
                inside -> android.graphics.Color.argb(58, 255, 72, 72)
                else -> android.graphics.Color.argb(24, 255, 72, 72)
            }
        }
    }

    val result = Bitmap.createBitmap(out, width, height, Bitmap.Config.ARGB_8888)
    if (workBitmap !== depthBitmap) workBitmap.recycle()
    return result
}

private fun createFocusGuideBitmap(depthBitmap: Bitmap, params: BlurPreviewParams): Bitmap {
    val workBitmap = scaleGuideSource(depthBitmap, 640)
    val width = workBitmap.width
    val height = workBitmap.height
    val pixels = IntArray(width * height)
    workBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
    
    val focus = params.focusDepth.coerceIn(0f, 255f) / 255f
    val falloff = 0.035f + params.blurFalloff.coerceIn(0f, 1f) * 0.72f
    val out = IntArray(width * height)

    for (i in pixels.indices) {
        val d = ((pixels[i] shr 16) and 0xFF) / 255f
        val dist = kotlin.math.abs(d - focus)
        
        // Red overlay that is opaque in focus and fades out in the transition zone
        if (dist < falloff) {
            val alpha = (1.0f - (dist / falloff)).coerceIn(0f, 1f)
            val redAlpha = (alpha * 120).toInt() // Max 120/255 opacity
            out[i] = android.graphics.Color.argb(redAlpha, 255, 40, 40)
        }
    }

    val result = Bitmap.createBitmap(out, width, height, Bitmap.Config.ARGB_8888)
    if (workBitmap !== depthBitmap) workBitmap.recycle()
    return result
}

private fun scaleGuideSource(bitmap: Bitmap, maxDimension: Int): Bitmap {
    val scale = min(1f, maxDimension.toFloat() / maxOf(bitmap.width, bitmap.height).toFloat())
    if (scale >= 1f) return bitmap
    return Bitmap.createScaledBitmap(
        bitmap,
        maxOf(1, (bitmap.width * scale).toInt()),
        maxOf(1, (bitmap.height * scale).toInt()),
        true
    )
}

private val bitmapRecycleHandler by lazy { Handler(Looper.getMainLooper()) }

private fun Bitmap?.replaceWith(next: Bitmap): Bitmap? {
    val previous = this
    if (previous != null && previous !== next && !previous.isRecycled) {
        bitmapRecycleHandler.postDelayed(
            {
                if (!previous.isRecycled) {
                    previous.recycle()
                }
            },
            650L
        )
    }
    return next
}

@Composable
private fun AnimeLoadingEffect() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.25f))
    ) {
        Box(
            modifier = Modifier
                .align(Alignment.Center)
                .clip(RoundedCornerShape(12.dp))
                .background(Color.Black.copy(alpha = 0.6f))
                .padding(horizontal = 20.dp, vertical = 12.dp)
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    strokeWidth = 2.5.dp,
                    color = AccentAmber
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    "Analyzing Depth...",
                    color = AccentAmber,
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
            }
        }
    }
}

@Composable
private fun SavingLoadingEffect() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.6f)),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(12.dp))
                .background(Color.Black.copy(alpha = 0.85f))
                .padding(horizontal = 32.dp, vertical = 20.dp)
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator(
                    modifier = Modifier.size(28.dp),
                    strokeWidth = 3.dp,
                    color = AccentAmber
                )
                Spacer(modifier = Modifier.height(14.dp))
                Text(
                    "Saving Image...",
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 0.5.sp
                )
            }
        }
    }
}
