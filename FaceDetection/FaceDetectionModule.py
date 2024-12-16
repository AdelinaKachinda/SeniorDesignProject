import cv2
import mediapipe as mp
import time

class FaceDetector:
    """
    A class for detecting faces in images or video using Mediapipe.
    """

    def __init__(self, minDetectionCon=0.5):
        """
        Initializes the FaceDetector with a minimum detection confidence.

        Parameters:
        minDetectionCon (float): Minimum confidence threshold for detections.
        """
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.FaceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, image, draw=True):
        """
        Detects faces in an image and optionally draws bounding boxes.

        Parameters:
        image (numpy.ndarray): Input image in BGR format.
        draw (bool): Whether to draw bounding boxes on the image.

        Returns:
        image (numpy.ndarray): The image with bounding boxes drawn (if `draw` is True).
        bboxs (list): A list of bounding boxes and confidence scores for detected faces.
        """
        # Convert the image to RGB as Mediapipe requires RGB input
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.FaceDetection.process(imgRGB)
        
        bboxs = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                # Extract bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape  # Image dimensions
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                bboxs.append([id, bbox, detection.score])  # Add to bounding boxes list
                
                if draw:
                    self.fancyDraw(image, bbox)  # Draw fancy bounding box
                    confidence = int(detection.score[0] * 100)  # Convert to percentage
                    cv2.putText(image, f'{confidence}%', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        
        return image, bboxs
    
    def fancyDraw(self, image, bbox, l=30, t=10):
        """
        Draws a stylized bounding box with lines and corners around a face.

        Parameters:
        image (numpy.ndarray): Input image.
        bbox (tuple): Bounding box coordinates (x, y, w, h).
        l (int): Line length for corners.
        t (int): Line thickness for corners.
        """
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Draw the main rectangle
        cv2.rectangle(image, bbox, (255, 0, 255), 2)

        # Draw stylized corners
        # Top left corner
        cv2.line(image, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(image, (x, y), (x, y + l), (255, 0, 255), t)

        # Top right corner
        cv2.line(image, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(image, (x1, y), (x1, y + l), (255, 0, 255), t)

        # Bottom left corner
        cv2.line(image, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(image, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # Bottom right corner
        cv2.line(image, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(image, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return image


def main():
    """
    Main function to demonstrate the FaceDetector class.
    """
    # Initialize video capture object
    cap = cv2.VideoCapture("FaceDetection/Videos/FD.MP4")
    if not cap.isOpened():
        print("Error: Video file not found or unable to open.")
        return
    
    pTime = 0  # Previous time for FPS calculation
    detector = FaceDetector()  # Create a FaceDetector object

    while True:
        success, image = cap.read()  # Read a frame from the video
        if not success:
            print("Reached the end of the video or cannot read the frame.")
            break
        
        # Detect faces and get bounding boxes
        image, bboxs = detector.findFaces(image)
        print(bboxs)  # Print detected bounding boxes
        
        # Calculate and display FPS
        cTime = time.time()  # Current time
        fps = 1 / (cTime - pTime)  # Frames per second
        pTime = cTime  # Update previous time

        cv2.putText(image, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        # Display the image
        cv2.imshow("Image", image)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
