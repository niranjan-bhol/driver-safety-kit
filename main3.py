import cv2
import face_recognition
import os
import numpy as np

# === Load Known Driver ===
KNOWN_FACES_DIR = "known_faces"
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)

# === Start Webcam ===
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_DUPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    identity = "Unknown"
    status = "Unknown"
    color = (0, 0, 255)  # Red

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if True in matches:
            best_match_index = np.argmin(face_distances)
            identity = known_names[best_match_index]
            status = "Authenticated"
            color = (0, 255, 0)  # Green

    # === Draw Black Panel ===
    cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), -1)  # Black box

    # === Display Driver Info ===
    cv2.putText(frame, f"Driver: {identity}", (10, 30), font, 0.7, color, 1)
    cv2.putText(frame, status, (10, 60), font, 0.7, color, 1)

    cv2.imshow("Face Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
