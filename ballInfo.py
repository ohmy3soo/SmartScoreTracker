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

join = []
check = []

whiteM = deque(maxlen=3)
yellowM = deque(maxlen=3)
redM = deque(maxlen=3)

whiteQ = deque()
yellowQ = deque()
redQ = deque()

whiteR = 0
yellowR = 0
redR = 0

whiteP = deque()
yellowP = deque()
redP = deque()

ROI_SIZE = 8

move = {'red': redM, 'yellow': yellowM, 'white': whiteM}
queue = {'red': redQ, 'yellow': yellowQ, 'white': whiteQ}
radius = {'red': redR, 'yellow': yellowR, 'white': whiteR}
pyr = {'red': redP, 'yellow': yellowP, 'white': whiteP}

width = height = 0
pw = ph = 0


def getDistance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def init(w, h, p_w, p_h):
    global width, height, pw, ph
    width = w
    height = h
    pw = p_w
    ph = p_h


def traceBall(color, frame, display):
    global radius, whiteR, redR, yellowR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = colorBoundary['lower_'+color]
    upper = colorBoundary['upper_'+color]

    colorImage = cv2.inRange(hsv, lower, upper)

    kernal = np.ones((3, 3), "uint8")
    colorImage = cv2.dilate(colorImage, kernal)
    pre_h = pre_w = 0
    if move[color][0] != -1:
        pre_h = ROI_SIZE * radius[color]
        pre_w = ROI_SIZE * radius[color]

        h1 = max(0, int(queue[color][0][1] - ROI_SIZE * radius[color]))
        if h1 == 0:
            pre_h = queue[color][0][1]
        h2 = min(int(queue[color][0][1] + ROI_SIZE * radius[color]), colorImage.shape[0])

        w1 = max(0, int(queue[color][0][0] - ROI_SIZE * radius[color]))
        if w1 == 0:
            pre_w = queue[color][0][0]
        w2 = min(int(queue[color][0][0] + ROI_SIZE * radius[color]), colorImage.shape[1])

        colorImage = colorImage[h1: h2, w1: w2]
    '''
    if color == 'yellow':
        cv2.imshow("ROI_" + color, colorImage)
        cv2.moveWindow("ROI_"+color, 612, 0)
    if color == 'white':
        cv2.imshow("ROI_" + color, colorImage)
        cv2.moveWindow("ROI_"+color, 612, 190)
    if color == 'red':
        cv2.imshow("ROI_" + color, colorImage)
        cv2.moveWindow("ROI_"+color, 612, 360)
    '''
    numOfLabels, img_label, stats, centroids \
        = cv2.connectedComponentsWithStats(colorImage)

    minD = 2 * frame.shape[1]
    idx = 0
    update = False

    for pic, centroid in enumerate(centroids):
        if stats[pic][0] == 0 and stats[pic][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        x, y, width, height, area = stats[pic]

        if 500 > area > 50:
            if width / height > 3 or height / width > 3:
                continue
            centerX, centerY = int(centroid[0]), int(centroid[1])

            d = getDistance(pre_h, pre_w, centerX, centerY)
            if d < minD:
                minD = d
                idx = pic
                update = True

    if update:
        x, y, width, height, area = stats[idx]
        if move[color][0] != -1:
            x = int(x - pre_w + queue[color][0][0])
            y = int(y - pre_h + queue[color][0][1])
        centerX = int(x + width/2)
        centerY = int(y + height/2)

        pre = queue[color][0]
        d = getDistance(pre[0], pre[1], centerX, centerY)
        if color != 'red':
            move[color].appendleft(d)

        queue[color].appendleft((centerX, centerY))
        if color == 'red':
            getPyrDistance(2, colorImage, color, pre_w, pre_h)
        if display:
            cv2.rectangle(frame, (x, y), (x + width, y + height), colors[color], 2)
            cv2.putText(frame, color, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[color])

    else:
        move[color].appendleft(-1)


def getPyrDistance(size, frame, color, w, h):
    global prePyr
    frame = cv2.pyrUp(frame)
    numOfLabels, img_label, stats, centroids \
        = cv2.connectedComponentsWithStats(frame)
    for pic, centroid in enumerate(centroids):
        if stats[pic][0] == 0 and stats[pic][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        x, y, width, height, area = stats[pic]
        x = int(x - w*size + queue[color][0][0]*size)
        y = int(y - h*size + queue[color][0][1]*size)
        # 공 객체 후보
        if 1000 > area > 200:
            pyr[color].appendleft((x, y))

    ddd = getDistance(pyr[color][1][0], pyr[color][1][1], pyr[color][0][0], pyr[color][0][1])
    move[color].appendleft(ddd)


def findBall(color, frame):
    global radius, whiteR, redR, yellowR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = colorBoundary['lower_'+color]
    upper = colorBoundary['upper_'+color]

    colorImage = cv2.inRange(hsv, lower, upper)

    kernal = np.ones((3, 3), "uint8")
    colorImage = cv2.dilate(colorImage, kernal)

    numOfLabels, img_label, stats, centroids \
        = cv2.connectedComponentsWithStats(colorImage)

    for pic, centroid in enumerate(centroids):
        if stats[pic][0] == 0 and stats[pic][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue

        x, y, w, h, area = stats[pic]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if 250 > area > 50:
            if x >= pw/2 and x <= frame.shape[1] - pw/2 and y >= ph/2 and y <= frame.shape[0] - ph/2:
                if w/h < 1.2 and h/w < 1.2:
                    print(color, 'rX: ', (w / 2))
                    print(color, 'rY: ', (h / 2))
                    radius[color] = (w + h) / 4
                    queue[color].appendleft((centerX, centerY))
                    pyr[color].appendleft((2*centerX, 2*centerY))
                    move[color].appendleft(0.0)
