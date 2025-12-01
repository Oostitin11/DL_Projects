// Server-Sent Events untuk real-time updates
import { type NextRequest, NextResponse } from "next/server"

export async function GET(req: NextRequest) {
  // Create a TransformStream untuk streaming detection results
  const encoder = new TextEncoder()

  const customReadable = new ReadableStream({
    async start(controller) {
      try {
        // Connect ke Python backend untuk stream detection results
        const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || "http://localhost:5000"

        const response = await fetch(`${pythonBackendUrl}/api/stream`)

        if (!response.ok) throw new Error("Stream connection failed")

        const reader = response.body?.getReader()
        if (!reader) throw new Error("No response body")

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          controller.enqueue(value)
        }
      } catch (error) {
        controller.error(error)
      } finally {
        controller.close()
      }
    },
  })

  return new NextResponse(customReadable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  })
}
