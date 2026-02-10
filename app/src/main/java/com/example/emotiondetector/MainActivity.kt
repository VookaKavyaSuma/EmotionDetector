package com.example.emotiondetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var tvEmotion: TextView
    private lateinit var overlayView: OverlayView
    private lateinit var cameraExecutor: ExecutorService
    private var faceLandmarker: FaceLandmarker? = null

    // Frame-skipping flag: prevents queueing while MediaPipe is busy
    @Volatile
    private var isProcessing = false

    // Monotonically increasing timestamp for MediaPipe (avoids duplicate-ts errors)
    private var lastTimestampMs = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraExecutor = Executors.newSingleThreadExecutor()

        viewFinder = findViewById(R.id.viewFinder)
        tvEmotion = findViewById(R.id.tvEmotion)
        overlayView = findViewById(R.id.overlayView)

        val switchOverlay = findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchOverlay)
        switchOverlay.setOnCheckedChangeListener { _, isChecked ->
            overlayView.visibility = if (isChecked) android.view.View.VISIBLE else android.view.View.GONE
        }

        setupFaceLandmarker()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupFaceLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()

        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumFaces(1)
            .setOutputFaceBlendshapes(true)
            .setResultListener { result, _ ->
                // Release processing lock so the next frame can be sent
                isProcessing = false
                processFaceResult(result)
            }
            .setErrorListener { error ->
                isProcessing = false
                Log.e("EmotionDetector", "AI Error: ${error.message}")
            }
            .build()

        try {
            faceLandmarker = FaceLandmarker.createFromOptions(this, options)
        } catch (e: Exception) {
            Log.e("EmotionDetector", "AI Failed to load: ${e.message}")
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(viewFinder.surfaceProvider) }

            // Low resolution + RGBA format = fastest possible bitmap path
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(480, 640))
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image -> processImage(image) }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("EmotionDetector", "Camera failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        // Skip frame if MediaPipe is still processing the previous one
        if (isProcessing) {
            imageProxy.close()
            return
        }

        // Fast bitmap from RGBA buffer (no YUV→RGB conversion overhead)
        val bitmap: Bitmap
        try {
            val plane = imageProxy.planes[0]
            val buffer = plane.buffer
            // rowStride/pixelStride gives actual row width in pixels (handles padding)
            val rowWidth = plane.rowStride / plane.pixelStride
            bitmap = Bitmap.createBitmap(rowWidth, imageProxy.height, Bitmap.Config.ARGB_8888)
            buffer.rewind()
            bitmap.copyPixelsFromBuffer(buffer)
        } catch (e: Exception) {
            imageProxy.close()
            return
        }

        // Use camera-reported rotation to orient the image upright for detection
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        imageProxy.close() // Free camera buffer ASAP

        val processedBitmap = if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
        } else {
            bitmap
        }

        // Tell overlay the dimensions of the (rotated) analysis image
        overlayView.setSourceInfo(processedBitmap.width, processedBitmap.height)

        // Guarantee monotonically increasing timestamps
        val now = System.currentTimeMillis()
        lastTimestampMs = if (now > lastTimestampMs) now else lastTimestampMs + 1

        isProcessing = true
        val mpImage = BitmapImageBuilder(processedBitmap).build()
        faceLandmarker?.detectAsync(mpImage, lastTimestampMs)
    }

    // ---------- Emotion Classification ----------

    private fun processFaceResult(result: FaceLandmarkerResult) {
        if (result.faceLandmarks().isEmpty()) {
            runOnUiThread {
                tvEmotion.text = "No Face Detected"
                overlayView.setResults(result)
            }
            return
        }

        val blendshapes = result.faceBlendshapes()
        if (!blendshapes.isPresent || blendshapes.get().isEmpty()) {
            runOnUiThread { tvEmotion.text = "Face Detected (No Blendshapes)" }
            return
        }

        val faceBlendshapes = blendshapes.get()[0]

        fun getScore(categoryName: String): Float {
            return faceBlendshapes.firstOrNull { it.categoryName() == categoryName }?.score() ?: 0f
        }

        val smileScore = (getScore("mouthSmileLeft") + getScore("mouthSmileRight")) / 2f
        val angryScore = (getScore("browDownLeft") + getScore("browDownRight")) / 2f
        val surpriseScore = (getScore("browInnerUp") + getScore("jawOpen")) / 2f
        val blinkScore = (getScore("eyeBlinkLeft") + getScore("eyeBlinkRight")) / 2f

        var emotion = "Neutral 😐"
        var confidence = 0f

        if (blinkScore > 0.6) {
            emotion = "Blinking / Sleepy 😴"; confidence = blinkScore
        } else if (smileScore > 0.4) {
            emotion = "Happy 😊"; confidence = smileScore
        } else if (angryScore > 0.4) {
            emotion = "Angry 😠"; confidence = angryScore
        } else if (surpriseScore > 0.4) {
            emotion = "Surprised 😲"; confidence = surpriseScore
        } else {
            emotion = "Neutral 😐"
            confidence = 1.0f - (smileScore + angryScore + surpriseScore)
        }

        val statusText = "Emotion: $emotion\nConfidence: ${String.format("%.0f%%", confidence * 100)}"

        runOnUiThread {
            tvEmotion.text = statusText
            overlayView.setResults(result)
        }
    }

    // ---------- Permissions ----------

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) startCamera()
            else Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
        }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceLandmarker?.close()
    }
}