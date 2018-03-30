import cv2
import numpy as np
from colorLabeler import ColorLabeler
from collections import OrderedDict
from collections import deque

white = deque()
yellow = deque()
red = deque()
balls = {"white":white, "yellow":yellow, "red":red}
minThreshold = 50
maxThreshold = 200

GLOBAL_MIDDLE_BOARD = (1270, 2540)
DOMESTIC_MIDDLE_BOARD = (1224, 2448)
h = int(DOMESTIC_MIDDLE_BOARD[0]/4)
w = int(DOMESTIC_MIDDLE_BOARD[1]/4)

M = None
R = 0
colors = OrderedDict({
            "red":(0,0,255),
            "yellow":(0,255,255),
            "white":(255,255,255),
            #"green":(0,255,0),
            "blue":(255,0,0) })


def setMatrix(image):
    board, mask = usingBlueBoard(image)
    pts1 = np.float32(board)
    pts2 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    global M
    M = cv2.getPerspectiveTransform(pts1, pts2)


def getWarp(image):
    return cv2.warpPerspective(image, M, (w, h))


def usingBlueBoard(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    blue_masked = cv2.bitwise_and(hsv, hsv, mask=mask_blue)
    blue_masked = cv2.cvtColor(blue_masked, cv2.COLOR_RGB2GRAY)

    cnts = cv2.findContours(blue_masked.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    board_w = 0;
    for (i, c) in enumerate(cnts):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) == 4:
            #cv2.drawContours(image, approx, -1, (0, 255, 0), 3)
            (x, y, w, h) = cv2.boundingRect(approx)
            if board_w < w:
                board_w = w
                board = approx
    return board, blue_masked


def findBalls(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([30, 50, 0])
    upper_blue = np.array([150, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow("mask!!", mask)
    #mask = cv2.dilate(mask, None, iterations=2)
    #mask = cv2.erode(mask, None, iterations=2)

    global R
    ballCount = 0
    cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    centers = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)

        if len(approx) > 5:
            ((x, y), radius) = cv2.minEnclosingCircle(approx)

            if 15 > radius > 5:
                R += radius
                ballCount += 1

                cl = ColorLabeler()
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                color = cl.label(lab, c)


                balls[color].appendleft((int(x),int(y)))
                drawLine(image, color)
                text = "({},{})".format(int(x),int(y))
                cv2.circle(image, (int(x), int(y)), int(radius), colors[color], 2)
                cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[color], 1)
                centers.append((x,y))
    R = R/ballCount
    #checkCollision(centers, radius)
    return image


def checkCollision(centers, r):
    '''
    for i, c1 in enumerate(centers):
        for j, c2 in enumerate(centers):
            if j>i:
                d = getDistance(c1,c2)
                print(i, j, int(d), int((2*r))**2)
                if d <= (2*r)**2:
                    print("Coliision!")
    '''
    #print()

def getDistance(c1, c2):
    #print(c1, c2)
    d = (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2
    return d


def onChange(x):
    pass


def drawLine(img, color):
    pts = balls[color]
    color = colors[color]
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        cv2.line(img, pts[i-1], pts[i], color, 1)

