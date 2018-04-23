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

whiteC = 0
yellowC = 0
redC = 0


ROI_SIZE = 8

queue = {'red': redQ, 'yellow': yellowQ, 'white': whiteQ}
radius = {'red': redR, 'yellow': yellowR, 'white': whiteR}
center = {'red': redC, 'yellow': yellowC, 'white':whiteC}

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


check = []

def traceBall(color, frame):
    global radius, whiteR, redR, yellowR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = colorBoundary['lower_'+color]
    upper = colorBoundary['upper_'+color]

    colorImage = cv2.inRange(hsv, lower, upper)

    kernal = np.ones((3, 3), "uint8")
    colorImage = cv2.dilate(colorImage, kernal)

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

    #print(h1, h2, w1, w2)
    colorImage = colorImage[h1: h2, w1: w2]

    #print(colorImage.shape)
    cv2.imshow("ROI_"+color, colorImage)


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

        # 공 객체 후보
        if 300 > area > 50:
            centerX, centerY = int(centroid[0]), int(centroid[1])

            d = getDistance(pre_h, pre_w, centerX, centerY)
            if d < minD:
                minD = d
                idx = pic
                # 찾음
                update = True
    # 찾음
    if update:
        x, y, width, height, area = stats[idx]
        #x = int(x - pre_w + queue[color][0][0])
        #y = int(y - pre_h + queue[color][0][1])
        x = int(x - pre_w + queue[color][0][0])
        y = int(y - pre_h + queue[color][0][1])
        centerX = int(x + width/2)
        centerY = int(y + height/2)

        pre = queue[color][0]
        rM = getDistance(pre[0], pre[1], centerX, centerY)
        '''
        if color == 'yellow':
            print('-'*10)
            print(pre[0],pre[1])
            print(centerX, centerY)
            print('-' * 10)
        '''
        '''
        if rM > 1.5:
            queue[color].appendleft((centerX, centerY))
            check.append(color)
        else:
            queue[color].appendleft(queue[color][0])
            rM = 0
        '''
        queue[color].appendleft((centerX, centerY))
        #cv2.circle(frame, (centerX, centerY), int(radius[color]), colors[color], 1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), colors[color], 2)
        cv2.putText(frame, color, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[color])
        return rM

    return -1





# 맨 처음 프레임에서 공을 찾는다. 초기위치 설정!
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

        if 300 > area > 50:
            if x >= pw/2 and x <= frame.shape[1] - pw/2 and y >= ph/2 and y <= frame.shape[0] - ph/2:
                if w/h < 1.2 or h/w < 1.2:
                    print(color, 'rX: ', (w / 2))
                    print(color, 'rY: ', (w / 2))

                    radius[color] = (w + h) / 4
                    queue[color].appendleft((centerX, centerY))
                    center[color] = (ROI_SIZE*radius[color], ROI_SIZE*radius[color])
                    print(radius[color], center[color])