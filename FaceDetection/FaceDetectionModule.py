import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.FaceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.FaceDetection.process(imgRGB)
        
        bboxs = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                bboxs.append([id, bbox, detection.score])
                if draw:
                    self.fancyDraw(image, bbox)
               
        
        return image, bboxs
    
    def fancyDraw(self, image, bbox, l=30, t = 10):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(image, bbox, (255, 0, 255), 2)
        # Top Left
        cv2.line(image, (x,y), (x+l, y), (255, 0, 255), t)
        cv2.line(image, (x,y), (x, y+l), (255, 0, 255), t)

        # Top right
        cv2.line(image, (x1,y), (x1-l, y), (255, 0, 255), t)
        cv2.line(image, (x1,y), (x1, y+l), (255, 0, 255), t)

        # Bottom left
        cv2.line(image, (x,y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(image, (x,y1), (x, y1-l), (255, 0, 255), t)

        # Bottom right
        cv2.line(image, (x1,y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(image, (x1,y1), (x1, y1-l), (255, 0, 255), t)

        return image



def main():
    # Ensure the video file path is correct
    cap = cv2.VideoCapture("FaceDetection/Videos/FD.MP4")
    if not cap.isOpened():
        print("Error: Video file not found or unable to open.")
        return
    
    pTime = 0
    detector = FaceDetector()

    while True:
        success, image = cap.read()
        if not success:
            print("Reached the end of the video or cannot read the frame.")
            break
        
        image, bboxs = detector.findFaces(image)
        print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
