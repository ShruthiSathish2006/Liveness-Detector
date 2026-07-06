import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = np.std(gray)
    noise = cv2.Laplacian(gray, cv2.CV_64F).std()

    
    if sharpness > 300:
        quality = "Excellent"
        color = (0, 255, 0)
    elif sharpness > 150:
        quality = "Good"
        color = (0, 255, 255)
    elif sharpness > 80:
        quality = "Fair"
        color = (0, 165, 255)
    else:
        quality = "Poor"
        color = (0, 0, 255)

    
    cv2.putText(frame, f"Quality: {quality}", (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Sharpness: {sharpness:.1f}", (10, 65),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Brightness: {brightness:.1f}", (10, 95),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Contrast: {contrast:.1f}", (10, 125),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Noise: {noise:.1f}", (10, 155),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Sample Quality Analyzer", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()