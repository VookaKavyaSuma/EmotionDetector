package com.example.emotiondetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.ProgressBar
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
import kotlin.math.abs
import kotlin.math.atan2

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var cameraExecutor: ExecutorService
    private var faceLandmarker: FaceLandmarker? = null

    // UI Elements
    private lateinit var tvEmoji: TextView
    private lateinit var tvEmotionLabel: TextView
    private lateinit var tvConfidenceText: TextView
    private lateinit var progressConfidence: ProgressBar
    private lateinit var tvYaw: TextView
    private lateinit var tvPitch: TextView
    private lateinit var tvRoll: TextView
    private lateinit var bottomHud: View

    @Volatile
    private var isProcessing = false
    private var lastTimestampMs = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize UI
        viewFinder = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.overlayView)
        tvEmoji = findViewById(R.id.tvEmoji)
        tvEmotionLabel = findViewById(R.id.tvEmotionLabel)
        tvConfidenceText = findViewById(R.id.tvConfidenceText)
        progressConfidence = findViewById(R.id.progressConfidence)
        tvYaw = findViewById(R.id.tvYaw)
        tvPitch = findViewById(R.id.tvPitch)
        tvRoll = findViewById(R.id.tvRoll)
        bottomHud = findViewById(R.id.bottomHud)

        // Setup the Overlay Toggle Switch (optional if you have one)
        try {
            val switchOverlay = findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.switchOverlay)
            switchOverlay.setOnCheckedChangeListener { _, isChecked ->
                overlayView.visibility = if (isChecked) View.VISIBLE else View.GONE
            }
        } catch (e: Exception) {
            // Ignore if switch is missing in layout
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
            .setOutputFaceBlendshapes(true) // Crucial for "Happy", "Wink" detection
            .setResultListener { result, _ ->
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
        if (isProcessing) {
            imageProxy.close()
            return
        }

        // 1. Safe Bitmap Conversion
        val bitmap: Bitmap
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        try {
            val plane = imageProxy.planes[0]
            val buffer = plane.buffer
            val rowWidth = plane.rowStride / plane.pixelStride
            val height = imageProxy.height
            val width = rowWidth

            // Create bitmap (this is a raw copy, might be padded)
            val tempBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            buffer.rewind()
            tempBitmap.copyPixelsFromBuffer(buffer)

            // Crop to actual size if needed
            bitmap = Bitmap.createBitmap(tempBitmap, 0, 0, imageProxy.width, imageProxy.height)
        } catch (e: Exception) {
            imageProxy.close()
            return
        }

        // Close imageProxy immediately after extracting data
        imageProxy.close()

        // 2. Rotate Bitmap to match screen
        val processedBitmap = if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
        } else {
            bitmap
        }

        // 3. Update Overlay Size
        runOnUiThread {
            overlayView.setSourceInfo(processedBitmap.width, processedBitmap.height)
        }

        // 4. Send to MediaPipe
        val now = System.currentTimeMillis()
        // Ensure timestamp is strictly increasing
        lastTimestampMs = if (now > lastTimestampMs) now else lastTimestampMs + 1

        isProcessing = true
        val mpImage = BitmapImageBuilder(processedBitmap).build()
        faceLandmarker?.detectAsync(mpImage, lastTimestampMs)
    }

    private fun processFaceResult(result: FaceLandmarkerResult) {
        // IMPORTANT: Always free the processing flag
        isProcessing = false

        if (result.faceLandmarks().isEmpty()) {
            runOnUiThread {
                tvEmotionLabel.text = "Searching..."
                tvEmoji.text = "🔍"
                tvConfidenceText.text = "No Face Detected"
                progressConfidence.progress = 0
                resetPose()
                overlayView.setResults(result)
            }
            return
        }

        // --- 1. Emotion Logic (Using Blendshapes - The "Fancy" Way) ---
        val blendshapes = result.faceBlendshapes()
        var emotion = "Neutral"
        var emoji = "😐"
        var confidence = 0f

        if (blendshapes.isPresent && blendshapes.get().isNotEmpty()) {
            val faceBlendshapes = blendshapes.get()[0]

            fun getScore(categoryName: String): Float {
                return faceBlendshapes.firstOrNull { it.categoryName() == categoryName }?.score() ?: 0f
            }

            val smileScore = (getScore("mouthSmileLeft") + getScore("mouthSmileRight")) / 2f
            val angryScore = (getScore("browDownLeft") + getScore("browDownRight")) / 2f
            val surpriseScore = (getScore("browInnerUp") + getScore("jawOpen")) / 2f
            val blinkScore = (getScore("eyeBlinkLeft") + getScore("eyeBlinkRight")) / 2f

            if (blinkScore > 0.5) {
                emotion = "Sleepy/Blinking"; emoji = "😴"; confidence = blinkScore
            } else if (smileScore > 0.4) {
                emotion = "Happy"; emoji = "😊"; confidence = smileScore
            } else if (angryScore > 0.4) {
                emotion = "Angry"; emoji = "😠"; confidence = angryScore
            } else if (surpriseScore > 0.3) {
                emotion = "Surprised"; emoji = "😲"; confidence = surpriseScore
            } else {
                emotion = "Neutral"; emoji = "😐"
                confidence = 1.0f - (smileScore + angryScore + surpriseScore)
                if (confidence < 0) confidence = 0f
            }
        }

        // --- 2. Head Pose Logic (Using Geometry - The "Reliable" Way) ---
        val landmarks = result.faceLandmarks()[0]

        // Key Landmarks
        val noseTip = landmarks[1]
        val leftCheek = landmarks[454]
        val rightCheek = landmarks[234]
        val leftEye = landmarks[33]
        val rightEye = landmarks[263]

        // YAW (Turn Left/Right)
        // Compare distance from nose to left cheek vs right cheek
        val distToLeft = abs(noseTip.x() - leftCheek.x())
        val distToRight = abs(rightCheek.x() - noseTip.x())
        val yawRatio = distToLeft / (distToRight + 0.001f) // Avoid divide by zero

        // Convert Ratio to Degrees (Approximation for display)
        // Neutral ratio is ~1.0. Left < 0.5, Right > 2.0
        var yawDeg = 0
        if (yawRatio < 0.8) yawDeg = -30 // Turning Left
        if (yawRatio > 1.2) yawDeg = 30  // Turning Right
        // Interpolate for smoother numbers
        yawDeg = ((yawRatio - 1.0) * 45).toInt()


        // PITCH (Look Up/Down)
        // Compare Nose Y to Eye center Y
        val eyeCenterY = (leftEye.y() + rightEye.y()) / 2.0f
        val noseToEyeDist = noseTip.y() - eyeCenterY
        // Normal distance is ~0.15. Larger = Down, Smaller = Up
        var pitchDeg = ((noseToEyeDist - 0.15) * 400).toInt()


        // ROLL (Tilt Head)
        // Compare Left Eye Y vs Right Eye Y
        val eyeDiffY = rightEye.y() - leftEye.y()
        val eyeDistX = rightEye.x() - leftEye.x()
        val rollRad = atan2(eyeDiffY, eyeDistX)
        val rollDeg = Math.toDegrees(rollRad.toDouble()).toInt()


        // --- 3. Update UI ---
        runOnUiThread {
            tvEmoji.text = emoji
            tvEmotionLabel.text = emotion
            tvConfidenceText.text = "Confidence: ${String.format("%.0f%%", confidence * 100)}"
            progressConfidence.progress = (confidence * 100).toInt()

            // Update the Buttons with real numbers now!
            tvYaw.text = "Y: ${yawDeg}°"
            tvPitch.text = "P: ${pitchDeg}°"
            tvRoll.text = "R: ${rollDeg}°"

            overlayView.setResults(result)
        }
    }

    private fun resetPose() {
        tvYaw.text = "Y: —"
        tvPitch.text = "P: —"
        tvRoll.text = "R: —"
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) startCamera()
            else Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
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