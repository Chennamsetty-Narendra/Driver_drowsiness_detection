import cv2
import dlib
import numpy as np
import winsound  # Import winsound for beep sound on Windows

# Load pre-trained models for face and landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define constants for sleep detection thresholds
EAR_THRESHOLD = 0.25
MOUTH_OPEN_THRESHOLD = 0.7
YAWN_FRAMES = 20
DROWSY_FRAMES = 30

# Counters for states
yawn_counter = 0
drowsy_counter = 0

# Car speed control (this is a simulation placeholder)
car_speed = 100  # Assume 100 is the normal speed

def calculate_ear(eye):
    # Compute the Eye Aspect Ratio (EAR)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mouth_open(mouth):
    # Compute the mouth opening ratio
    A = np.linalg.norm(mouth[3] - mouth[9])  # vertical distance
    C = np.linalg.norm(mouth[0] - mouth[6])  # horizontal distance
    mouth_open_ratio = A / C
    return mouth_open_ratio

def control_car_speed(state):
    global car_speed
    if state == "ACTIVE":
        car_speed = 100  # Normal speed
    elif state == "YAWNING":
        car_speed = 80  # Reduce speed
        winsound.Beep(1000, 500)  # Play beep sound (1000 Hz, 500 ms)
    elif state == "DROWSY" or state == "SLEEPY":
        car_speed = 40  # Significantly reduce speed or stop the car
        winsound.Beep(1000, 500)  # Play beep sound (1000 Hz, 500 ms)
    print(f"Driver State: {state}, Car Speed: {car_speed}")

def detect_driver_state(frame):
    global yawn_counter, drowsy_counter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract the eye coordinates
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:60]

        # Calculate EAR and mouth open ratio
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mouth_open_ratio = calculate_mouth_open(mouth)

        if ear < EAR_THRESHOLD:
            drowsy_counter += 1
            if drowsy_counter >= DROWSY_FRAMES:
                control_car_speed("DROWSY")
        elif mouth_open_ratio > MOUTH_OPEN_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= YAWN_FRAMES:
                control_car_speed("YAWNING")
        else:
            drowsy_counter = 0
            yawn_counter = 0
            control_car_speed("ACTIVE")

# Main loop to process video stream
cap = cv2.VideoCapture(0)  # Webcam feed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_driver_state(frame)

    cv2.imshow('Driver State Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
