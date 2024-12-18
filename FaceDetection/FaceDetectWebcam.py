import cv2
import time

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit the program.")

# Initialize the time for FPS calculation
prev_time = time.time()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image with stricter parameters for accuracy
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,    # Slightly higher scale factor for more precise detections
        minNeighbors=8,     # Increase minNeighbors to reduce false positives
        minSize=(50, 50)    # Increase minSize for detecting larger, clearer faces
    )

    # Ensure only the largest face is detected
    if len(faces) > 0:
        # Select the face with the largest bounding box area
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face

        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Calculate Frames Per Second (FPS)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display FPS above the bounding box
        cv2.putText(frame, f'FPS: {int(fps)}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)

    # Display the frame with face detection and FPS
    cv2.imshow('Face Detection with FPS', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
