import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import pygame
import threading

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize pygame mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("emergency-alarm.mp3")
alarm_on = False
eyes_closed_start = None

# Function to play alarm in a loop
def play_alarm():
    global alarm_on
    while alarm_on:
        if not pygame.mixer.get_busy():
            alarm_sound.play()
        time.sleep(1)

# Eye aspect ratio calculation
def eye_aspect_ratio(landmarks, eye_points, w, h):
    p1 = np.array([int(landmarks[eye_points[0]].x * w), int(landmarks[eye_points[0]].y * h)])
    p2 = np.array([int(landmarks[eye_points[1]].x * w), int(landmarks[eye_points[1]].y * h)])
    p3 = np.array([int(landmarks[eye_points[2]].x * w), int(landmarks[eye_points[2]].y * h)])
    p4 = np.array([int(landmarks[eye_points[3]].x * w), int(landmarks[eye_points[3]].y * h)])
    p5 = np.array([int(landmarks[eye_points[4]].x * w), int(landmarks[eye_points[4]].y * h)])
    p6 = np.array([int(landmarks[eye_points[5]].x * w), int(landmarks[eye_points[5]].y * h)])

    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# Start video capture
cap = cv2.VideoCapture(0)

blink_times = deque()
yawn_times = deque()
blink_start_time = time.time()
yawn_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    h, w, _ = frame.shape
    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_points = [33, 160, 158, 133, 153, 144]
            right_eye_points = [362, 385, 387, 263, 373, 380]

            left_ear = eye_aspect_ratio(face_landmarks.landmark, left_eye_points, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, right_eye_points, w, h)

            eye_threshold = 0.22

            left_status = "open" if left_ear > eye_threshold else "close"
            right_status = "open" if right_ear > eye_threshold else "close"

            left_color = (255, 0, 0) if left_status == "open" else (0, 0, 255)
            right_color = (255, 0, 0) if right_status == "open" else (0, 0, 255)

            cv2.putText(frame, f"LEFT EYE  : {left_status.upper()}", (30, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, left_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"RIGHT EYE : {right_status.upper()}", (30, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, right_color, 1, cv2.LINE_AA)

            # Handle alarm logic based on eye closure duration
            if left_status == "close" and right_status == "close":
                if not eyes_closed_start:
                    eyes_closed_start = current_time
                elif current_time - eyes_closed_start >= 5:
                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(target=play_alarm, daemon=True).start()
                    cv2.putText(frame, "Alert", (30, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            else:
                eyes_closed_start = None
                if alarm_on:
                    alarm_on = False
                    pygame.mixer.stop()

            # Blink Tracking
            if left_status == "close" and right_status == "close":
                if current_time - blink_start_time > 0.2:
                    blink_times.append(current_time)
                    blink_start_time = current_time

            while blink_times and current_time - blink_times[0] > 60:
                blink_times.popleft()

            blink_count = len(blink_times)
            blink_text = f"{blink_count} times / NORMAL"
            blink_color = (0, 0, 0)

            if blink_count > 15:
                blink_text = f"{blink_count} times / SLEEPY"
                blink_color = (0, 0, 255)

            cv2.putText(frame, f"Blinks : {blink_text}", (30, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, blink_color, 1, cv2.LINE_AA)

            # Yawn Detection
            mouth_points = [
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61
            ]

            lip_distance = abs(face_landmarks.landmark[13].y - face_landmarks.landmark[14].y)
            yawn_threshold = 0.06

            if lip_distance > yawn_threshold:
                if not yawn_active:
                    yawn_active = True
                    yawn_times.append((current_time, 0))
                yawn_times[-1] = (yawn_times[-1][0], yawn_times[-1][1] + (1 / 30))
            else:
                yawn_active = False

            while yawn_times and current_time - yawn_times[0][0] > 60:
                yawn_times.popleft()

            yawn_duration = sum(duration for _, duration in yawn_times)
            yawn_text = f"{round(yawn_duration, 1)} sec / NORMAL"
            yawn_color = (0, 0, 0)

            if yawn_duration > 10:
                yawn_text = f"{round(yawn_duration, 1)} sec / SLEEPY"
                yawn_color = (0, 0, 255)

            cv2.putText(frame, f"Yawns : {yawn_text}", (30, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, yawn_color, 1, cv2.LINE_AA)

            for i in range(len(mouth_points) - 1):
                pt1 = (int(face_landmarks.landmark[mouth_points[i]].x * w), int(face_landmarks.landmark[mouth_points[i]].y * h))
                pt2 = (int(face_landmarks.landmark[mouth_points[i + 1]].x * w), int(face_landmarks.landmark[mouth_points[i + 1]].y * h))
                cv2.line(frame, pt1, pt2, (0, 165, 255), 2)

            for eye in [left_eye_points, right_eye_points]:
                for idx in eye:
                    x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
