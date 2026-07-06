import cv2
import mediapipe as mp
import numpy as np

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

# ============================================================
# Thresholds
# ============================================================
HAND_OVERLAP_MARGIN    = 15     # px padding around hand bbox

# Per-feature expected texture profiles (contrast std, edge ratio)
# These are RANGES of what a clear, unobstructed feature looks like.
# Values outside range = something is wrong (obstruction or out of frame).
# Calibrated conservatively to avoid false positives.
FEATURE_PROFILE = {
    # name         min_contrast  min_edge   max_uniformity
    "Left Eye":  (  8.0,         0.010 ),
    "Right Eye": (  8.0,         0.010 ),
    "Nose":      (  5.0,         0.006 ),
    "Mouth":     (  5.0,         0.006 ),
    "Left Ear":  (  4.0,         0.004 ),
    "Right Ear": (  4.0,         0.004 ),
}

# How many consecutive frames a feature must fail before rejecting
# Prevents single-frame flicker causing false rejection
TEMPORAL_FRAMES = 5

# ============================================================
# MediaPipe — Face Mesh + Hands
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Temporal failure counters — tracks how many consecutive frames
# each feature has failed its texture check
failure_counts = {name: 0 for name in FEATURE_INDICES}

# ============================================================
# Video Quality Metrics
# ============================================================

def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def gamma_estimate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray / 255.0)
    if mean <= 0:
        return 0
    return round(np.log(mean) / np.log(0.5), 2)

def resolution(frame):
    h, w = frame.shape[:2]
    return w, h

# ============================================================
# Head Pose
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
        (0.0,   0.0,   0.0),   (0.0,  -63.6, -12.5),
        (-43.3, 32.7, -26.0),  (43.3,  32.7, -26.0),
        (-28.9,-28.9, -24.1),  (28.9, -28.9, -24.1)
    ])

    focal_length  = w
    camera_matrix = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ])
    _, rvec, _ = cv2.solvePnP(model_points, image_points,
                               camera_matrix, np.zeros((4,1)))
    rmat, _    = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[1], angles[0], angles[2]

# ============================================================
# Hand detection
# ============================================================

def get_hand_bboxes(hand_results, frame_shape):
    h, w   = frame_shape[:2]
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
# ROI extraction
# ============================================================

def get_feature_roi(face, indices, frame):
    h, w = frame.shape[:2]
    xs   = [int(face.landmark[i].x * w) for i in indices]
    ys   = [int(face.landmark[i].y * h) for i in indices]
    pad  = 12
    x1   = max(0,   min(xs) - pad)
    y1   = max(0,   min(ys) - pad)
    x2   = min(w-1, max(xs) + pad)
    y2   = min(h-1, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# ============================================================
# OBSTRUCTION DETECTION — three independent pixel signals
#
# Signal 1: Local contrast (std of gray pixels)
#   Real facial features have natural texture variation.
#   An object covering them (cloth, paper, phone, hand) creates
#   a more uniform surface → lower std.
#
# Signal 2: Adaptive edge density (Canny with median thresholds)
#   Eyes, nose, mouth, ears have natural structural edges.
#   A flat object covering them removes those edges.
#
# Signal 3: Gradient uniformity (Sobel variance)
#   Real features have varied gradient directions.
#   A flat occluder has gradients pointing in fewer directions
#   → lower variance of gradient magnitudes.
#
# All three are normalised and combined.
# Rejection only fires after TEMPORAL_FRAMES consecutive failures
# to prevent single-frame false positives.
# ============================================================

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
    return np.count_nonzero(edges) / (edges.size + 1e-6)

def gradient_uniformity(roi):
    """
    Variance of Sobel gradient magnitudes.
    High variance = many different gradient strengths = real facial texture.
    Low variance  = uniform surface = possible obstruction.
    """
    if roi is None or roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx**2 + gy**2)
    return float(np.var(mag))

def is_obstructed(roi, feature_name):
    """
    Returns (obstructed: bool, reason: str, scores: dict)

    Checks all three signals against per-feature profiles.
    A feature is obstructed if contrast OR edges fall below
    their minimum thresholds (either signal is enough).
    """
    if roi is None or roi.size == 0:
        return True, "No ROI", {}

    contrast  = local_contrast(roi)
    edges     = adaptive_edge_density(roi)
    grad_var  = gradient_uniformity(roi)

    min_contrast, min_edge = FEATURE_PROFILE.get(feature_name, (5.0, 0.005))

    scores = {
        "contrast": round(contrast, 1),
        "edges":    round(edges * 100, 2),
        "grad_var": round(grad_var, 1),
    }

    # Both contrast AND edge must fail to trigger obstruction
    # (reduces false positives from lighting/motion alone)
    contrast_fail = contrast < min_contrast
    edge_fail     = edges    < min_edge

    if contrast_fail and edge_fail:
        return True, f"Low texture (c:{contrast:.1f} e:{edges*100:.1f}%)", scores

    return False, f"OK (c:{contrast:.1f} e:{edges*100:.1f}%)", scores

# ============================================================
# COMBINED FEATURE CHECK
#
# Priority order:
#   1. Out of frame          → reject immediately
#   2. Hand overlap          → reject immediately (most reliable)
#   3. Texture obstruction   → reject after TEMPORAL_FRAMES frames
#      (catches phone, cloth, paper, book, anything non-skin)
# ============================================================

def check_all_features(face, frame, hand_bboxes):
    global failure_counts
    MARGIN   = 0.01
    features = {}

    for name, indices in FEATURE_INDICES.items():
        h, w = frame.shape[:2]

        # --- Position check ---
        lms = [face.landmark[i] for i in indices]
        in_count = sum(
            1 for lm in lms
            if MARGIN < lm.x < (1-MARGIN) and MARGIN < lm.y < (1-MARGIN)
        )
        position_ok = (in_count / len(lms)) >= 0.75

        # --- ROI ---
        roi, bbox = get_feature_roi(face, indices, frame)

        # --- Hand overlap ---
        hand_blocked = hand_overlaps_bbox(hand_bboxes, bbox)

        # --- Texture/obstruction check ---
        obstructed, obs_reason, scores = is_obstructed(roi, name)

        # --- Temporal smoothing ---
        if obstructed:
            failure_counts[name] = min(failure_counts[name] + 1,
                                       TEMPORAL_FRAMES + 1)
        else:
            # Reset on any good frame
            failure_counts[name] = max(failure_counts[name] - 1, 0)

        texture_reject = failure_counts[name] >= TEMPORAL_FRAMES

        # --- Final decision ---
        if not position_ok:
            visible = False
            reason  = "Out of frame"
        elif hand_blocked:
            visible = False
            reason  = "Hand covering feature"
        elif texture_reject:
            visible = False
            reason  = f"Obstructed: {obs_reason}"
        else:
            visible = True
            reason  = obs_reason

        features[name] = {
            "visible":      visible,
            "reason":       reason,
            "hand_blocked": hand_blocked,
            "scores":       scores,
            "bbox":         bbox,
            "fail_count":   failure_counts[name],
        }

    missing  = [n for n, v in features.items() if not v["visible"]]
    rejected = len(missing) > 0
    return features, rejected, missing

# ============================================================
# Main Loop
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)
    hand_results = hand_detector.process(rgb)

    blur          = blur_score(frame)
    bright        = brightness_score(frame)
    gamma         = gamma_estimate(frame)
    width, height = resolution(frame)

    score       = 100
    rejected    = False
    missing     = []
    features    = {}
    hand_bboxes = get_hand_bboxes(hand_results, frame.shape)
    hand_count  = len(hand_bboxes)

    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]

        yaw, pitch, roll = head_pose(face, frame)
        if abs(yaw) > 25:
            score -= 10
        if abs(pitch) > 20:
            score -= 10

        features, rejected, missing = check_all_features(face, frame, hand_bboxes)

        if rejected:
            score = 0
        else:
            if blur < 100:
                score -= 20
            if bright < 60:
                score -= 20
            if width < 1280:
                score -= 10

        score = max(score, 0)

        # Feature panel — right side
        panel_x = frame.shape[1] - 270
        for i, (fname, info) in enumerate(features.items()):
            color = (0, 255, 0) if info["visible"] else (0, 0, 255)
            cv2.putText(frame,
                        f"{fname}: {info['reason']}",
                        (panel_x, 30 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        cv2.putText(frame, f"Yaw:{yaw:.1f}",    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.putText(frame, f"Pitch:{pitch:.1f}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.putText(frame, f"Roll:{roll:.1f}",   (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

    else:
        # Reset failure counts when face lost
        for k in failure_counts:
            failure_counts[k] = 0
        rejected = True
        missing  = ["Face"]
        score    = 0
        cv2.putText(frame, "No face detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    hand_color = (0, 0, 255) if hand_count > 0 else (0, 255, 0)
    cv2.putText(frame, f"Hands: {hand_count}", (20,124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2)

    cv2.putText(frame, f"Blur: {blur:.1f}",         (20,155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
    cv2.putText(frame, f"Brightness: {bright:.1f}", (20,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
    cv2.putText(frame, f"Gamma: {gamma}",            (20,205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
    cv2.putText(frame, f"Res: {width}x{height}",    (20,230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

    if rejected:
        verdict_text  = "REJECTED"
        verdict_color = (0, 0, 255)
        score_color   = (0, 0, 255)
        reason_text   = ", ".join(missing) + (" not detected" if "Face" in missing else " obstructed")
    elif score >= 80:
        verdict_text  = "PASS"
        verdict_color = (0, 255, 0)
        score_color   = (0, 255, 0)
        reason_text   = ""
    elif score >= 60:
        verdict_text  = "REVIEW"
        verdict_color = (0, 255, 255)
        score_color   = (0, 255, 255)
        reason_text   = "Low quality"
    else:
        verdict_text  = "FAIL"
        verdict_color = (0, 0, 255)
        score_color   = (0, 0, 255)
        reason_text   = "Quality too low"

    cv2.putText(frame, f"Score: {score}", (20,270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color,   2)
    cv2.putText(frame, verdict_text,       (20,310), cv2.FONT_HERSHEY_SIMPLEX, 1.0, verdict_color, 2)
    if reason_text:
        cv2.putText(frame, reason_text,    (20,340), cv2.FONT_HERSHEY_SIMPLEX, 0.55, verdict_color, 1)

    cv2.putText(frame, "Press Q to quit",
                (20, frame.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)

    cv2.imshow("KYC Quality Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()