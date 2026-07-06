import cv2
import numpy as np
import asyncio
import websockets
import json
import base64
import time

# ── Config ────────────────────────────────────────────────────────────────────
HOST = "localhost"
PORT = 8765
TARGET_FPS = 15          # analysis + frame send rate
JPEG_QUALITY = 60        # lower = smaller frames, less bandwidth

connected_clients = set()

# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))
    noise      = float(cv2.Laplacian(gray, cv2.CV_64F).std())

    if sharpness > 150:
        quality = "EXCELLENT"
    elif sharpness > 100:
        quality = "GOOD"
    elif sharpness > 80:
        quality = "FAIR"
    else:
        quality = "POOR"

    h, w = frame.shape[:2]

    # Encode frame as JPEG → base64
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    frame_b64 = base64.b64encode(buf).decode('utf-8')

    return {
        "sharpness":  round(float(sharpness), 1),
        "brightness": round(brightness, 1),
        "contrast":   round(contrast, 1),
        "noise":      round(noise, 1),
        "quality":    quality,
        "width":      w,
        "height":     h,
        "frame":      frame_b64,
    }

# ── WebSocket handler ─────────────────────────────────────────────────────────
async def handler(ws):
    print(f"[+] Client connected: {ws.remote_address}")
    connected_clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        connected_clients.discard(ws)
        print(f"[-] Client disconnected")

# ── Capture loop ──────────────────────────────────────────────────────────────
async def capture_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    interval = 1.0 / TARGET_FPS
    frame_count = 0
    print(f"[*] Webcam open — streaming at {TARGET_FPS} fps on ws://{HOST}:{PORT}")

    try:
        while True:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(interval)
                continue

            frame_count += 1
            if connected_clients:
                data = analyze(frame)
                data["frame_count"] = frame_count
                msg = json.dumps(data)
                # broadcast to all connected clients
                await asyncio.gather(
                    *[ws.send(msg) for ws in list(connected_clients)],
                    return_exceptions=True
                )

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, interval - elapsed))
    finally:
        cap.release()
        print("[*] Camera released")

# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    async with websockets.serve(handler, HOST, PORT):
        await capture_loop()

if __name__ == "__main__":
    asyncio.run(main())
