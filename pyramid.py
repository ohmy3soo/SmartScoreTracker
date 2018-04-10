import cv2
import numpy as np
import imutils
import billiardFunction
import math
import ballInfo
from collections import deque

pw = 18
ph = 16

def drawLines(img):
    cv2.line(img, (pw, ph), (img.shape[1]-pw, ph), (0,0,255), 1)
    cv2.line(img, (img.shape[1]-pw, ph), (img.shape[1]-pw, img.shape[0]-ph), (0, 0, 255), 1)
    cv2.line(img, (img.shape[1]-pw, img.shape[0]-ph), (pw, img.shape[0]-ph), (0, 0, 255), 1)
    cv2.line(img, (pw, img.shape[0]-ph), (pw, ph), (0, 0, 255), 1)

BGRcolor = {"red":(0,0,255),
            "yellow":(0,255,255),
            "white":(255,255,255),
            "green":(0,255,0),
            "blue":(255,0,0) }

join = []

camera = cv2.VideoCapture("/Users/kihunahn/Desktop/videoSrc/1.mp4")
ret, img = camera.read()
img = imutils.resize(img, width=600)

billiardFunction.setMatrix(img)
s = []
p1 = 'yellow'
p2 = 'white'
r = 'red'


while True:
    '''
    p1 = 'yellow'
    p2 = 'white'
    r = 'red'
    '''
    ret, img = camera.read()
    img = imutils.resize(img, width=600)

    frame = billiardFunction.getWarp(img)
    cv2.imshow('1', frame)
    for i in range(2):
        frame = cv2.pyrDown(frame)

    #for i in range(2):
    #    frame = cv2.pyrUp(frame)

    cv2.imshow('2', frame)


    rV = ballInfo.traceBall(r, frame)
    p1V = ballInfo.traceBall(p1, frame)
    p2V = ballInfo.traceBall(p2, frame)


    cv2.imshow('3', frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
#out.release()
camera.release()
cv2.destroyAllWindows()