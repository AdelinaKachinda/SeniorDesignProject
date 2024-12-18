import cv2
import mediapipe as mp
import time

# Initialize webcam capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

pTime = 0  # Initialize previous time for FPS calculation

# Initialize Mediapipe Face Mesh and drawing specifications
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)  # Limit to one face
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)  # Drawing style

print("Press 'q' to quit the program.")

while True:
    success, img = cap.read()  # Read a frame from the webcam
    if not success:
        print("Error: Could not read a frame from the webcam.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Mediapipe
    results = faceMesh.process(imgRGB)  # Process the frame for face landmarks

    # If face landmarks are detected, draw them on the frame
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            # Draw the face mesh on the frame
            mpDraw.draw_landmarks(img, faceLm, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

            # Debugging: Print landmark coordinates
            for lm in faceLm.landmark:
                print(lm)

    # Calculate Frames Per Second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # Display FPS on the video
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Show the frame in a window
    cv2.imshow("Webcam Face Mesh", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
