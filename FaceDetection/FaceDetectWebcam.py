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

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,               # Input image
        scaleFactor=1.1,    # Scale factor for image pyramid
        minNeighbors=5,     # Minimum neighbors for a valid detection
        minSize=(30, 30)    # Minimum size of detected faces
    )

    # Calculate Frames Per Second (FPS)
    current_time = time.time()  # Current time
    fps = 1 / (current_time - prev_time)  # FPS = 1 / time taken for one frame
    prev_time = current_time  # Update previous time

    # Draw rectangles around detected faces and display FPS
    for (x, y, w, h) in faces:
        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

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
