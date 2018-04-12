import cv2
import numpy as np
import math
from collections import deque
from collections import OrderedDict

colorBoundary = {'lower_white': np.array([0, 0, 240]),
                 'upper_white': np.array([255, 15, 255]),
                 'lower_yellow': np.array([15, 60, 200]),
                 'upper_yellow': np.array([60, 255, 255]),
                 #'lower_yellow': np.array([15, 139, 132]),
                 #'upper_yellow': np.array([75, 255, 198]),
                 'lower_red': np.array([160, 100, 111]),
                 'upper_red': np.array([180, 255, 255])}

colors = OrderedDict({
            "red":(0,0,255),
            "yellow":(0,255,255),
            "white":(255,255,255),
            #"green":(0,255,0),
            "blue":(255,0,0) })


whiteQ = deque()
yellowQ = deque()
redQ = deque()
rSide = deque()
lSide = deque()
getV = deque()
''''''
whiteR = 0
yellowR = 0
redR = 0



queue = {'red': redQ, 'yellow': yellowQ, 'white': whiteQ}
radius = {'red': redR, 'yellow': yellowR, 'white': whiteR}


width = 0
height = 0
pw = 18
ph = 16

def getDistance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def setInit(img):
    width = img.shape[1]
    height = img.shape[0]
    pw = 18
    ph = 16


def traceBall(color, frame):
    global radius, whiteR, redR, yellowR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = colorBoundary['lower_'+color]
    upper = colorBoundary['upper_'+color]

    colorImage = cv2.inRange(hsv, lower, upper)

    kernal = np.ones((3, 3), "uint8")
    colorImage = cv2.dilate(colorImage, kernal)

    contours = cv2.findContours(colorImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    minD = 2 * frame.shape[1]
    idx = 0
    update = False

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if 200 > area > 60:

            x, y, w, h = cv2.boundingRect(contour)
            if x >= pw/2 and x <= frame.shape[1]-pw and y >= ph/2 and y <= frame.shape[0]-ph:
                centerX = int(x + w / 2)
                centerY = int(y + h / 2)

                if not queue[color]:
                    print(color, 'rX: ', (w / 2))
                    print(color, 'rY: ', (h / 2))
                    radius[color] = (w+h)/4
                    print(radius[color])
                    queue[color].appendleft((centerX, centerY))

                d = getDistance(queue[color][0][0], queue[color][0][1], centerX, centerY)
                if d < minD:
                    minD = d
                    idx = pic
                    update = True

    if update:
        x, y, w, h = cv2.boundingRect(contours[idx])
        centerX = int(x + w / 2)
        centerY = int(y + h / 2)
        pre = queue[color][0]
        rM = getDistance(pre[0], pre[1], centerX, centerY)
        if rM > 1.5:
            queue[color].appendleft((int(centerX), int(centerY)))
        else:
            queue[color].appendleft(queue[color][0])

        rM = getDistance(queue[color][0][0], queue[color][0][1], queue[color][1][0], queue[color][1][1])

        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[color], 2)
        cv2.putText(frame, color, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[color])
        return rM

    return -1
