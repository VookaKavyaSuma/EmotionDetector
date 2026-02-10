package com.example.emotiondetector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import androidx.core.content.ContextCompat
import android.graphics.BlurMaskFilter

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: FaceLandmarkerResult? = null

    // Subtle dots for all 468 landmarks
    private val dotPaint = Paint().apply {
        color = ContextCompat.getColor(context!!, R.color.accentCyan)
        alpha = 180
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // Bright contour paint for key facial features (Eyes)
    private val eyesPaint = Paint().apply {
        color = ContextCompat.getColor(context!!, R.color.accentCyan)
        strokeWidth = 4f
        style = Paint.Style.STROKE
        isAntiAlias = true
        maskFilter = android.graphics.BlurMaskFilter(8f, android.graphics.BlurMaskFilter.Blur.SOLID)
    }

    // Lips
    private val lipsPaint = Paint().apply {
        color = ContextCompat.getColor(context!!, R.color.accentCoral)
        strokeWidth = 4f
        style = Paint.Style.STROKE
        isAntiAlias = true
        maskFilter = android.graphics.BlurMaskFilter(8f, android.graphics.BlurMaskFilter.Blur.SOLID)
    }

    // Eyebrows
    private val browsPaint = Paint().apply {
        color = ContextCompat.getColor(context!!, R.color.accentGold)
        strokeWidth = 4f
        style = Paint.Style.STROKE
        isAntiAlias = true
        maskFilter = android.graphics.BlurMaskFilter(8f, android.graphics.BlurMaskFilter.Blur.SOLID)
    }

    // Face Oval
    private val facePaint = Paint().apply {
        color = Color.WHITE
        alpha = 100
        strokeWidth = 2f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    @Volatile private var imageWidth = 1
    @Volatile private var imageHeight = 1

    fun setSourceInfo(width: Int, height: Int) {
        imageWidth = width
        imageHeight = height
    }

    fun setResults(faceLandmarkerResult: FaceLandmarkerResult) {
        results = faceLandmarkerResult
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val currentResults = results ?: return
        if (currentResults.faceLandmarks().isEmpty()) return

        val landmarks = currentResults.faceLandmarks()[0]

        // Match PreviewView's fillCenter (CenterCrop) scaling
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val imgW = imageWidth.toFloat()
        val imgH = imageHeight.toFloat()

        val scaleFactor = kotlin.math.max(viewW / imgW, viewH / imgH)
        val offsetX = (viewW - imgW * scaleFactor) / 2f
        val offsetY = (viewH - imgH * scaleFactor) / 2f

        // Pre-compute all screen positions
        val sx = FloatArray(landmarks.size)
        val sy = FloatArray(landmarks.size)
        for (i in landmarks.indices) {
            val mirroredX = 1.0f - landmarks[i].x()
            sx[i] = mirroredX * imgW * scaleFactor + offsetX
            sy[i] = landmarks[i].y() * imgH * scaleFactor + offsetY
        }

        // 1. Draw all 468 landmarks (subtle dots)
        for (i in landmarks.indices) {
            // Draw every 5th landmark to reduce visual clutter, or all if preferred
            if (i % 3 == 0) canvas.drawCircle(sx[i], sy[i], 1.5f, dotPaint)
        }

        // 2. Draw contour lines for key facial features with specific colors
        drawConnections(canvas, sx, sy, landmarks.size, FACE_OVAL, facePaint)
        drawConnections(canvas, sx, sy, landmarks.size, LEFT_EYE, eyesPaint)
        drawConnections(canvas, sx, sy, landmarks.size, RIGHT_EYE, eyesPaint)
        drawConnections(canvas, sx, sy, landmarks.size, LIPS_OUTER, lipsPaint)
        drawConnections(canvas, sx, sy, landmarks.size, LIPS_INNER, lipsPaint)
        drawConnections(canvas, sx, sy, landmarks.size, LEFT_EYEBROW, browsPaint)
        drawConnections(canvas, sx, sy, landmarks.size, RIGHT_EYEBROW, browsPaint)
    }

    private fun drawConnections(
        canvas: Canvas, sx: FloatArray, sy: FloatArray,
        count: Int, connections: IntArray, paint: Paint
    ) {
        var i = 0
        while (i < connections.size - 1) {
            val s = connections[i]
            val e = connections[i + 1]
            if (s < count && e < count) {
                canvas.drawLine(sx[s], sy[s], sx[e], sy[e], paint)
            }
            i += 2
        }
    }

    companion object {
        // Standard MediaPipe 468-landmark connections, stored as flat [start,end, start,end, ...]
        // Face oval
        private val FACE_OVAL = intArrayOf(
            10,338, 338,297, 297,332, 332,284, 284,251, 251,389, 389,356, 356,454,
            454,323, 323,361, 361,288, 288,397, 397,365, 365,379, 379,378, 378,400,
            400,377, 377,152, 152,148, 148,176, 176,149, 149,150, 150,136, 136,172,
            172,58, 58,132, 132,93, 93,234, 234,127, 127,162, 162,21, 21,54,
            54,103, 103,67, 67,109, 109,10
        )
        // Left eye
        private val LEFT_EYE = intArrayOf(
            263,249, 249,390, 390,373, 373,374, 374,380, 380,381, 381,382, 382,362,
            263,466, 466,388, 388,387, 387,386, 386,385, 385,384, 384,398, 398,362
        )
        // Right eye
        private val RIGHT_EYE = intArrayOf(
            33,7, 7,163, 163,144, 144,145, 145,153, 153,154, 154,155, 155,133,
            33,246, 246,161, 161,160, 160,159, 159,158, 158,157, 157,173, 173,133
        )
        // Outer lips
        private val LIPS_OUTER = intArrayOf(
            61,146, 146,91, 91,181, 181,84, 84,17, 17,314, 314,405, 405,321,
            321,375, 375,291, 61,185, 185,40, 40,39, 39,37, 37,0, 0,267,
            267,269, 269,270, 270,409, 409,291
        )
        // Inner lips
        private val LIPS_INNER = intArrayOf(
            78,95, 95,88, 88,178, 178,87, 87,14, 14,317, 317,402, 402,318,
            318,324, 324,308, 78,191, 191,80, 80,81, 81,82, 82,13, 13,312,
            312,311, 311,310, 310,415, 415,308
        )
        // Left eyebrow
        private val LEFT_EYEBROW = intArrayOf(
            276,283, 283,282, 282,295, 295,285, 300,293, 293,334, 334,296, 296,336
        )
        // Right eyebrow
        private val RIGHT_EYEBROW = intArrayOf(
            46,53, 53,52, 52,65, 65,55, 70,63, 63,105, 105,66, 66,107
        )
    }
}