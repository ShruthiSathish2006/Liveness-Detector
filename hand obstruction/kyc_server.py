import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import json
import base64
import time

# ============================================================
# MediaPipe landmark index groups
# ============================================================
LEFT_EYE_IDX   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE_IDX       = [1, 2, 4, 5, 6, 19, 94, 168, 195, 197, 236, 354, 399, 420, 456]
MOUTH_IDX      = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LEFT_EAR_IDX   = [234, 227, 132, 136, 150, 176, 148]
RIGHT_EAR_IDX  = [454, 447, 361, 365, 379, 400, 377]

FEATURE_INDICES = {
    "Left Eye":  LEFT_EYE_IDX,
    "Right Eye": RIGHT_EYE_IDX,
    "Nose":      NOSE_IDX,
    "Mouth":     MOUTH_IDX,
    "Left Ear":  LEFT_EAR_IDX,
    "Right Ear": RIGHT_EAR_IDX,
}

HAND_OVERLAP_MARGIN = 15

FEATURE_PROFILE = {
    "Left Eye":  (8.0, 0.010),
    "Right Eye": (8.0, 0.010),
    "Nose":      (5.0, 0.006),
    "Mouth":     (5.0, 0.006),
    "Left Ear":  (4.0, 0.004),
    "Right Ear": (4.0, 0.004),
}

TEMPORAL_FRAMES = 5

HOST = "localhost"
PORT = 8765
TARGET_FPS = 15
JPEG_QUALITY = 65

# ============================================================
# MediaPipe setup
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
hand_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

failure_counts = {name: 0 for name in FEATURE_INDICES}
connected_clients = set()

# ============================================================
# Video quality
# ============================================================
def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def gamma_estimate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray / 255.0)
    if mean <= 0:
        return 0.0
    return round(float(np.log(mean) / np.log(0.5)), 2)

# ============================================================
# Head pose
# ============================================================
def head_pose(face_landmarks, frame):
    h, w = frame.shape[:2]
    image_points = np.array([
        (face_landmarks.landmark[1].x*w,   face_landmarks.landmark[1].y*h),
        (face_landmarks.landmark[152].x*w, face_landmarks.landmark[152].y*h),
        (face_landmarks.landmark[33].x*w,  face_landmarks.landmark[33].y*h),
        (face_landmarks.landmark[263].x*w, face_landmarks.landmark[263].y*h),
        (face_landmarks.landmark[61].x*w,  face_landmarks.landmark[61].y*h),
        (face_landmarks.landmark[291].x*w, face_landmarks.landmark[291].y*h)
    ], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
    ])
    focal_length  = w
    camera_matrix = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ])
    _, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4,1)))
    rmat, _    = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[1]), float(angles[0]), float(angles[2])

# ============================================================
# Hand detection
# ============================================================
def get_hand_bboxes(hand_results, frame_shape):
    h, w = frame_shape[:2]
    bboxes = []
    if not hand_results.multi_hand_landmarks:
        return bboxes
    for hand_lms in hand_results.multi_hand_landmarks:
        xs = [lm.x * w for lm in hand_lms.landmark]
        ys = [lm.y * h for lm in hand_lms.landmark]
        bboxes.append((
            int(min(xs)) - HAND_OVERLAP_MARGIN,
            int(min(ys)) - HAND_OVERLAP_MARGIN,
            int(max(xs)) + HAND_OVERLAP_MARGIN,
            int(max(ys)) + HAND_OVERLAP_MARGIN,
        ))
    return bboxes

def hand_overlaps_bbox(hand_bboxes, feature_bbox):
    fx1, fy1, fx2, fy2 = feature_bbox
    for (hx1, hy1, hx2, hy2) in hand_bboxes:
        if fx1 < hx2 and fx2 > hx1 and fy1 < hy2 and fy2 > hy1:
            return True
    return False

# ============================================================
# ROI + obstruction
# ============================================================
def get_feature_roi(face, indices, frame):
    h, w = frame.shape[:2]
    xs  = [int(face.landmark[i].x * w) for i in indices]
    ys  = [int(face.landmark[i].y * h) for i in indices]
    pad = 12
    x1  = max(0,   min(xs) - pad)
    y1  = max(0,   min(ys) - pad)
    x2  = min(w-1, max(xs) + pad)
    y2  = min(h-1, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def local_contrast(roi):
    if roi is None or roi.size == 0:
        return 0.0
    return float(np.std(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))

def adaptive_edge_density(roi):
    if roi is None or roi.size == 0:
        return 0.0
    gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    median = np.median(gray)
    lo     = max(0,   int(0.4 * median))
    hi     = min(255, int(1.1 * median))
    edges  = cv2.Canny(gray, lo, hi)
    return float(np.count_nonzero(edges) / (edges.size + 1e-6))

def gradient_uniformity(roi):
    if roi is None or roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx**2 + gy**2)
    return float(np.var(mag))

def is_obstructed(roi, feature_name):
    if roi is None or roi.size == 0:
        return True, "No ROI", {}
    contrast = local_contrast(roi)
    edges    = adaptive_edge_density(roi)
    grad_var = gradient_uniformity(roi)
    min_contrast, min_edge = FEATURE_PROFILE.get(feature_name, (5.0, 0.005))
    scores = {
        "contrast": round(contrast, 1),
        "edges":    round(edges * 100, 2),
        "grad_var": round(grad_var, 1),
    }
    if contrast < min_contrast and edges < min_edge:
        return True, f"Low texture", scores
    return False, "OK", scores

def check_all_features(face, frame, hand_bboxes):
    global failure_counts
    MARGIN   = 0.01
    features = {}
    for name, indices in FEATURE_INDICES.items():
        h, w = frame.shape[:2]
        lms  = [face.landmark[i] for i in indices]
        in_count = sum(
            1 for lm in lms
            if MARGIN < lm.x < (1-MARGIN) and MARGIN < lm.y < (1-MARGIN)
        )
        position_ok  = (in_count / len(lms)) >= 0.75
        roi, bbox    = get_feature_roi(face, indices, frame)
        hand_blocked = hand_overlaps_bbox(hand_bboxes, bbox)
        obstructed, obs_reason, scores = is_obstructed(roi, name)

        if obstructed:
            failure_counts[name] = min(failure_counts[name] + 1, TEMPORAL_FRAMES + 1)
        else:
            failure_counts[name] = max(failure_counts[name] - 1, 0)

        texture_reject = failure_counts[name] >= TEMPORAL_FRAMES

        if not position_ok:
            visible = False; reason = "Out of frame"
        elif hand_blocked:
            visible = False; reason = "Hand covering"
        elif texture_reject:
            visible = False; reason = "Obstructed"
        else:
            visible = True;  reason = "Clear"

        features[name] = {
            "visible":      visible,
            "reason":       reason,
            "hand_blocked": hand_blocked,
            "scores":       scores,
            "fail_count":   failure_counts[name],
        }

    missing  = [n for n, v in features.items() if not v["visible"]]
    rejected = len(missing) > 0
    return features, rejected, missing

# ============================================================
# WebSocket handler
# ============================================================
async def handler(ws):
    print(f"[+] Client connected: {ws.remote_address}")
    connected_clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        connected_clients.discard(ws)
        print(f"[-] Client disconnected")

# ============================================================
# Capture + analysis loop
# ============================================================
async def capture_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    interval    = 1.0 / TARGET_FPS
    frame_count = 0
    print(f"[*] KYC server running on ws://{HOST}:{PORT}")

    try:
        while True:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(interval)
                continue

            frame_count += 1

            if not connected_clients:
                await asyncio.sleep(interval)
                continue

            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            hand_results = hand_detector.process(rgb)

            blur   = round(blur_score(frame), 1)
            bright = round(brightness_score(frame), 1)
            gamma  = gamma_estimate(frame)
            h, w   = frame.shape[:2]

            hand_bboxes = get_hand_bboxes(hand_results, frame.shape)
            hand_count  = len(hand_bboxes)

            score    = 100
            rejected = False
            missing  = []
            features = {}
            yaw = pitch = roll = 0.0

            if face_results.multi_face_landmarks:
                face = face_results.multi_face_landmarks[0]
                yaw, pitch, roll = head_pose(face, frame)

                if abs(yaw) > 25:   score -= 10
                if abs(pitch) > 20: score -= 10

                features, rejected, missing = check_all_features(face, frame, hand_bboxes)

                if rejected:
                    score = 0
                else:
                    if blur   < 100:  score -= 20
                    if bright < 60:   score -= 20
                    if w      < 1280: score -= 10

                score = max(score, 0)
            else:
                for k in failure_counts:
                    failure_counts[k] = 0
                rejected = True
                missing  = ["Face"]
                score    = 0

            if rejected:
                verdict = "REJECTED"
            elif score >= 80:
                verdict = "PASS"
            elif score >= 60:
                verdict = "REVIEW"
            else:
                verdict = "FAIL"

            _, buf     = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_b64  = base64.b64encode(buf).decode('utf-8')

            payload = {
                "frame":       frame_b64,
                "frame_count": frame_count,
                "width":       w,
                "height":      h,
                "blur":        blur,
                "brightness":  bright,
                "gamma":       gamma,
                "hand_count":  hand_count,
                "yaw":         round(yaw,   1),
                "pitch":       round(pitch, 1),
                "roll":        round(roll,  1),
                "score":       score,
                "verdict":     verdict,
                "rejected":    rejected,
                "missing":     missing,
                "face_found":  bool(face_results.multi_face_landmarks),
                "features":    {
                    name: {
                        "visible":      info["visible"],
                        "reason":       info["reason"],
                        "hand_blocked": info["hand_blocked"],
                        "scores":       info["scores"],
                        "fail_count":   info["fail_count"],
                    }
                    for name, info in features.items()
                }
            }

            msg = json.dumps(payload)
            await asyncio.gather(
                *[ws.send(msg) for ws in list(connected_clients)],
                return_exceptions=True
            )

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, interval - elapsed))
    finally:
        cap.release()
        print("[*] Camera released")

async def main():
    async with websockets.serve(handler, HOST, PORT):
        await capture_loop()

if __name__ == "__main__":
    asyncio.run(main())
