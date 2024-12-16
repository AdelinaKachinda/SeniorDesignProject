import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture("FaceDetection/Videos/FD.MP4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDetection = mpFaceDetection.FaceDetection()

while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = FaceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin + ih), \
                    int(bboxC.width * iw), int(bboxC.height + ih)
            
            cv2.rectangle(image, bbox, (255, 0, 255), 2)
        

            
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255 , 0), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(1)
