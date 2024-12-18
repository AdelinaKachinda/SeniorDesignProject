import cv2
import mediapipe as mp
import time

# Initialize webcam capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Initialize the previous frame time for FPS calculation
pTime = 0

# Initialize Mediapipe face detection module and drawing utilities
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)  # Adjust confidence as needed

print("Press 'q' to quit the program.")

# Main loop to process webcam frames
while True:
    # Capture a frame from the webcam
    success, image = cap.read()
    if not success:
        print("Error: Could not read a frame from the webcam.")
        break

    # Convert the frame to RGB as Mediapipe requires RGB input
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection on the RGB frame
    results = faceDetection.process(imgRGB)

    # If faces are detected, draw bounding boxes and detection scores
    if results.detections:
        for id, detection in enumerate(results.detections):
            # Extract the bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = image.shape  # Get image dimensions
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw a rectangle around the detected face
            cv2.rectangle(image, bbox, (255, 0, 255), 2)

            # Display the detection confidence score above the bounding box
            confidence = int(detection.score[0] * 100)  # Convert to percentage
            cv2.putText(image, f'{confidence}%', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # Calculate Frames Per Second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # Display FPS on the video frame
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Show the processed frame
    cv2.imshow("Webcam Face Detection", image)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
