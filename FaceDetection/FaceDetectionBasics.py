import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture("Video/FaceDetectionVideo.mp4")

while True:
    success, image = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)