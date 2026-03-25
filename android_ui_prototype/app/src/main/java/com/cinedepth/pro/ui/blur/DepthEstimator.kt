package com.cinedepth.pro.ui.blur

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.common.ops.NormalizeOp
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class DepthEstimator(context: Context) {
    companion object {
        const val INPUT_IMAGE_SIZE = 518
    }
    private var interpreter: Interpreter? = null
    private val modelFileName = "Depth-Anything-V2.tflite"

    private val inputImageSize = INPUT_IMAGE_SIZE
    private var isInitialized = false

    // Pre-allocate for performance
    private val inputTensorImage = TensorImage(DataType.FLOAT32)
    private val outputBuffer = ByteBuffer.allocateDirect(1 * inputImageSize * inputImageSize * 4).apply {
        order(ByteOrder.nativeOrder())
    }

    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(0.0f, 255.0f)) // Native normalization is faster than Kotlin loops
        .build()

    init {
        try {
            val modelBuffer = loadModelFile(context, modelFileName)
            val compatList = CompatibilityList()
            val options = Interpreter.Options()

            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = GpuDelegate.Options().apply {
                    // FP16 precision is much faster on mobile GPUs with minimal quality loss for depth
                    setPrecisionLossAllowed(true)
                    setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                }
                val gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
            } else {
                options.setNumThreads(4)
            }

            interpreter = Interpreter(modelBuffer, options)
            isInitialized = true
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun isReady(): Boolean = isInitialized

    fun estimateDepth(bitmap: Bitmap): FloatArray? {
        if (!isInitialized || interpreter == null) return null

        // Optimized native image processing
        inputTensorImage.load(bitmap)
        val processedImage = imageProcessor.process(inputTensorImage)
        
        outputBuffer.rewind()

        try {
            interpreter?.run(processedImage.buffer, outputBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }

        outputBuffer.rewind()
        val floatArray = FloatArray(inputImageSize * inputImageSize)
        outputBuffer.asFloatBuffer().get(floatArray)

        // Optimized normalization loop
        var minV = Float.MAX_VALUE
        var maxV = Float.MIN_VALUE
        
        // Single pass for min/max
        for (i in floatArray.indices) {
            val v = floatArray[i]
            if (v < minV) minV = v
            if (v > maxV) maxV = v
        }
        
        val range = if (maxV - minV > 0) maxV - minV else 1f
        val invRange = 1.0f / range
        
        // Single pass for normalization
        for (i in floatArray.indices) {
            floatArray[i] = (floatArray[i] - minV) * invRange
        }

        return floatArray
    }

    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
