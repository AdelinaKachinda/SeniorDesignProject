import cv2
import mediapipe as mp
import time

# Initialize video capture object for the video file
cap = cv2.VideoCapture("FaceDetection/Videos/FD.MP4")

# Initialize the previous frame time to calculate FPS
pTime = 0

# Initialize Mediapipe face detection module and drawing utilities
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

# Main loop to process video frames
while True:
    # Read a frame from the video
    success, image = cap.read()
    if not success:
        print("End of video or unable to read frame.")
        break

    # Convert the frame to RGB as Mediapipe processes RGB images
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection on the RGB image
    results = faceDetection.process(imgRGB)

    # Debugging: Print detection results to console
    print(results)

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
            cv2.putText(image, f'{confidence}%', (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    # Calculate and display the Frames Per Second (FPS)
    cTime = time.time()  # Get current time
    fps = 1 / (cTime - pTime)  # Calculate FPS
    pTime = cTime  # Update previous time
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Display the processed frame in a window
    cv2.imshow("Image", image)

    # Wait for 1 millisecond and exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
