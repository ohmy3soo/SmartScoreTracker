import cv2
import numpy as np
import imutils
import billiardFunction
import ballInfo
from collections import deque
import time
videoPath = "/Users/kihunahn/Desktop/videoSrc/"
'''
videoList = ["1.mp4",
                "2.mp4",
                "3.mp4",
                "4.mp4",
                "hard1.mp4",
                "hard2.mp4",
                "hard3.mp4", ###
                "hard4.mp4"]
'''
videoList = ["1", "2", "3", "4", "hard1", "hard2", "hard3", "hard4"]
videoName = videoPath + videoList[4] + ".mp4"

camera = cv2.VideoCapture(videoName)
ret, img = camera.read()
print(img.shape)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Be sure to use the lower case
out = cv2.VideoWriter('/Users/kihunahn/Desktop/videoSrc/fps30/' + videoList[4]+'.avi', fourcc, 30.0, (img.shape[1], img.shape[0]))


while camera.isOpened():
    ret, img = camera.read()
    cv2.imshow('frame', img)
    key = cv2.waitKey(1)
    out.write(img)
    if key & 0xFF == ord('q'):
        break

out.release()
camera.release()
cv2.destroyAllWindows()