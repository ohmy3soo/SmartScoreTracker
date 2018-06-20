import cv2
import numpy as np
from collections import OrderedDict

minThreshold = 50
maxThreshold = 200

GLOBAL_MIDDLE_BOARD = (1270, 2540)
DOMESTIC_MIDDLE_BOARD = (1224, 2448)
h = int(DOMESTIC_MIDDLE_BOARD[0]/4)
w = int(DOMESTIC_MIDDLE_BOARD[1]/4)

M = None

colors = OrderedDict({
            "red":(0,0,255),
            "yellow":(0,255,255),
            "white":(255,255,255),
            #"green":(0,255,0),
            "blue":(255,0,0) })


def setMatrix(image):
    board, mask = usingBoard(image)
    pts1 = np.float32(board)
    pts2 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    global M
    M = cv2.getPerspectiveTransform(pts1, pts2)


def getWarp(image):
    return cv2.warpPerspective(image, M, (w, h))

'''
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
'''
def usingBoard(image, color="blue"):
    '''
    :param image:
        The image is a BGR frame of input video.

    :param color:
        The color means the color of board.
        The blue, green value is stored for use.
        default : "blue"

    :return board, mask:
        Returns a list of coordinate values of each vertex of the rectangle
        recognized by the billiard table.

        Returns the color mask of the pool table color to detect the pool table.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == "blue":
        lower = np.array([110, 100, 100])
        upper = np.array([130, 255, 255])
    elif color == "green":
        lower = np.array([50, 100, 100])
        upper = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

    cnts = cv2.findContours(masked.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
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
    return board, masked

