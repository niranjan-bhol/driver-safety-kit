import cv2
import mediapipe as mp
import time
import pygame  # For playing alarm sound

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize Mediapipe Face Detection for Bounding Box
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize Pygame for alarm sound
pygame.mixer.init()
alarm_file = "emergency-alarm.mp3"

# Start video capture
cap = cv2.VideoCapture(0)

# Track how long the driver is looking left or right
attention_start_time = None  
current_status = "Road"  
alarm_playing = False  # Track if alarm is currently playing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (Mediapipe requires RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face bounding box
    face_results = face_detection.process(frame_rgb)

    h, w, _ = frame.shape  # Get frame dimensions

    # Display title with extra spacing
    cv2.putText(frame, "Driver Attention and Safety Drive System", (30, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w_bbox, h_bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w_bbox, y + h_bbox), (0, 255, 0), 2)

            # Add "Face" label above the bounding box
            cv2.putText(frame, "Face", (x, y - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Detect facial landmarks for head position estimation
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get key points for head position estimation
            nose_tip = face_landmarks.landmark[1]   # Nose tip
            left_ear = face_landmarks.landmark[234]  # Left ear landmark
            right_ear = face_landmarks.landmark[454] # Right ear landmark

            # Convert relative coordinates to absolute
            nose_x = int(nose_tip.x * w)
            left_x = int(left_ear.x * w)
            right_x = int(right_ear.x * w)

            # Determine head direction
            if nose_x < left_x:  
                status = "Right"
                color = (0, 255, 255)  # Yellow
            elif nose_x > right_x:  
                status = "Left"
                color = (0, 255, 255)  # Yellow
            else:
                status = "Road"
                color = (0, 255, 0)  # Green

            # Display head position (Road/Left/Right)
            cv2.putText(frame, status, (30, 100), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2, cv2.LINE_AA)

            # Check if driver is looking away for more than 10 seconds
            if status in ["Left", "Right"]:
                if attention_start_time is None:
                    attention_start_time = time.time()
                elif time.time() - attention_start_time > 10:
                    # Display "Not Normal" with double spacing below status
                    cv2.putText(frame, "Not Normal", (30, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    # Display "Alert" below "Not Normal" while the alarm is ringing
                    cv2.putText(frame, "Alert", (30, 220), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    # Start alarm if not already playing
                    if not alarm_playing:
                        pygame.mixer.music.load(alarm_file)
                        pygame.mixer.music.play(-1)  # Loop alarm
                        alarm_playing = True
            else:
                # Reset timer and stop alarm when driver looks straight
                attention_start_time = None
                if alarm_playing:
                    pygame.mixer.music.stop()
                    alarm_playing = False

    # Show the frame
    cv2.imshow("Driver Attention Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
