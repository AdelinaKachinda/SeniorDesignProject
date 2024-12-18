import cv2
import mediapipe as mp
import time

# Initialize video capture from a file
cap = cv2.VideoCapture("FaceDetection/Videos/FD.MP4")

# Initialize previous time for FPS calculation
pTime = 0

# Initialize Mediapipe drawing utilities and Face Mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()  # Face Mesh detector; to detect multiple faces, adjust max_num_faces parameter
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)  # Style for face mesh landmarks

while True:
    # Read a frame from the video
    success, img = cap.read()
    if not success:
        print("End of video or error reading frame.")
        break

    # Convert frame to RGB as Mediapipe requires RGB input
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB frame with Face Mesh
    results = faceMesh.process(imgRGB)

    # If face landmarks are detected, draw them on the frame
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            # Draw the landmarks and connections on the frame
            mpDraw.draw_landmarks(img, faceLm, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

            # Loop through each landmark for debugging or additional processing
            for id, lm in enumerate(faceLm.landmark):
                # Convert normalized landmark coordinates to pixel values
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)  # Print landmark ID and its coordinates

    # Calculate Frames Per Second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # Display FPS on the video frame
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow("Image", img)

    # Exit loop when any key is pressed (you can change to 'q' for specific control)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
