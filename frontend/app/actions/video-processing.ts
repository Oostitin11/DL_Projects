"use server"

interface ProcessingResult {
  success: boolean
  stats?: {
    maleIn: number
    maleOut: number
    femaleIn: number
    femaleOut: number
    currentCount: number
    fps: number
    duration: number
  }
  error?: string
}

/**
 * Process video file dengan backend Python
 * Bisa digunakan sebagai server action dari client components
 */
export async function processVideoFile(formData: FormData): Promise<ProcessingResult> {
  try {
    const video = formData.get("video") as File

    if (!video) {
      return {
        success: false,
        error: "No video file provided",
      }
    }

    // Convert file to buffer
    const bytes = await video.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Get Python backend URL from environment
    const backendUrl = process.env.PYTHON_BACKEND_URL || "http://localhost:5000"

    // Send to Python backend
    const response = await fetch(`${backendUrl}/api/detect`, {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Length": buffer.length.toString(),
      },
      body: buffer,
      timeout: 300000, // 5 minute timeout untuk large videos
    })

    if (!response.ok) {
      const errorText = await response.text()
      return {
        success: false,
        error: `Backend error: ${response.statusText} - ${errorText}`,
      }
    }

    const result = await response.json()

    return {
      success: true,
      stats: {
        maleIn: result.maleIn || 0,
        maleOut: result.maleOut || 0,
        femaleIn: result.femaleIn || 0,
        femaleOut: result.femaleOut || 0,
        currentCount: result.currentCount || 0,
        fps: result.fps || 0,
        duration: result.duration || 0,
      },
    }
  } catch (error) {
    console.error("Video processing error:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    }
  }
}

/**
 * Start webcam streaming ke backend
 */
export async function startWebcamStream(): Promise<ProcessingResult> {
  try {
    const backendUrl = process.env.PYTHON_BACKEND_URL || "http://localhost:5000"

    const response = await fetch(`${backendUrl}/api/webcam-start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      return {
        success: false,
        error: "Failed to start webcam stream",
      }
    }

    return {
      success: true,
    }
  } catch (error) {
    console.error("Webcam start error:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    }
  }
}

/**
 * Validate backend connection
 */
export async function validateBackendConnection(): Promise<boolean> {
  try {
    const backendUrl = process.env.PYTHON_BACKEND_URL || "http://localhost:5000"

    const response = await fetch(`${backendUrl}/health`, {
      method: "GET",
      timeout: 5000,
    })

    return response.ok
  } catch (error) {
    console.error("Backend connection validation failed:", error)
    return false
  }
}
