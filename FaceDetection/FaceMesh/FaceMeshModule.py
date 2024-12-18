import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    """
    A class for detecting face mesh landmarks using Mediapipe.
    """

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        Initializes the FaceMeshDetector.

        Parameters:
        staticMode (bool): If True, processes input as a static image.
        maxFaces (int): Maximum number of faces to detect.
        minDetectionCon (float): Minimum confidence value for detection.
        minTrackCon (float): Minimum confidence value for tracking.
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        # Initialize Mediapipe Face Mesh and drawing utilities
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, 
                                                 max_num_faces=self.maxFaces, 
                                                 min_detection_confidence=self.minDetectionCon, 
                                                 min_tracking_confidence=self.minTrackCon)
        # Style for drawing landmarks
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Processes the image to detect face mesh landmarks.

        Parameters:
        img (numpy.ndarray): Input image.
        draw (bool): If True, draws face landmarks on the image.

        Returns:
        img (numpy.ndarray): Image with drawn landmarks (if draw=True).
        faces (list): List of face landmarks for each detected face.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for Mediapipe
        results = self.faceMesh.process(imgRGB)  # Process the image for face landmarks
        faces = []  # List to store face landmarks

        if results.multi_face_landmarks:  # If landmarks are detected
            for faceLm in results.multi_face_landmarks:
                if draw:  # Draw landmarks if enabled
                    self.mpDraw.draw_landmarks(img, faceLm, self.mpFaceMesh.FACEMESH_TESSELATION, 
                                               self.drawSpec, self.drawSpec)

                face = []  # List to store individual face landmarks
                for id, lm in enumerate(faceLm.landmark):
                    ih, iw, ic = img.shape  # Get image dimensions
                    x, y = int(lm.x * iw), int(lm.y * ih)  # Convert normalized coordinates to pixels
                    print(id, x, y)  # Debugging: Print landmark ID and coordinates
                    face.append([x, y])  # Append landmark coordinates

                faces.append(face)  # Append face landmarks to faces list

        return img, faces  # Return the image and detected faces


def main():
    """
    Main function to demonstrate the FaceMeshDetector.
    """
    # Initialize video capture (path to video file or webcam index)
    cap = cv2.VideoCapture("FaceDetection/Videos/FD.MP4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    pTime = 0  # Initialize previous time for FPS calculation
    detector = FaceMeshDetector()  # Create FaceMeshDetector instance

    while True:
        success, img = cap.read()  # Read a frame from the video
        if not success:
            print("Error: Could not read a frame from the video or end of video.")
            break

        img, faces = detector.findFaceMesh(img, True)  # Detect face mesh in the frame

        if len(faces) != 0:  # If faces are detected, print the count
            print(len(faces))

        # Calculate Frames Per Second (FPS)
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        # Display FPS on the video frame
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Display the frame with landmarks
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop when 'q' is pressed
            break

    # Release video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
