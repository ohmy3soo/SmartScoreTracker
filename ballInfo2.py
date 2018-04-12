import cv2
import numpy as np
import math
from collections import deque

colorBoundary = {'lower_white': np.array([0, 0, 240]),
                 'upper_white': np.array([255, 15, 255]),
                 'lower_yellow': np.array([15, 60, 200]),
                 'upper_yellow': np.array([60, 255, 255]),
                 #'lower_yellow': np.array([15, 139, 132]),
                 #'upper_yellow': np.array([75, 255, 198]),
                 'lower_red': np.array([160, 100, 111]),
                 'upper_red': np.array([180, 255, 255])}


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
pw = 0
ph = 0


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

        if 200 > area > 70:
            x, y, w, h = cv2.boundingRect(contour)
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
        queue[color].appendleft((int(centerX), int(centerY)))

        rM = getDistance(pre[0], pre[1], centerX, centerY)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, color, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
        return rM

    return -1


class Yellow:
    last_prediction = deque()
    last_measurement = deque()

    def KF(self, frame, x, y, draw=True):
        global current_measurement, measurements, last_measurement, current_prediction, last_prediction

        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if(len(self.last_prediction) >=2):
            d = getDistance(self.last_prediction[0][0], self.last_prediction[0][1], current_measurement[0],
                            current_measurement[1])
            print(d)

        self.kalman.correct(current_measurement)
        current_prediction = self.kalman.predict()

        self.last_prediction.appendleft(current_prediction)
        self.last_measurement.appendleft(current_measurement)

        if len(self.last_prediction) >= 2:
            for i in range(1, len(self.last_measurement)):
                cv2.line(frame, (self.last_measurement[i][0], self.last_measurement[i][1]),
                         (self.last_measurement[i-1][0], self.last_measurement[i-1][1]), (0, 255, 0), 1)

                cv2.line(frame, (self.last_prediction[i][0], self.last_prediction[i][1]),
                         (self.last_prediction[i-1][0], self.last_prediction[i-1][1]), (255, 255, 0), 1)


    kalman = cv2.KalmanFilter(4, 2, 1)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

class Red:

    last_prediction = deque()
    last_measurement = deque()

    def KF(self, frame, x, y, draw=True):
        global current_measurement, measurements, last_measurement, current_prediction, last_prediction

        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])

        self.kalman.correct(current_measurement)
        current_prediction = self.kalman.predict()

        self.last_prediction.appendleft(current_prediction)
        self.last_measurement.appendleft(current_measurement)

        if len(self.last_prediction) >= 2:
            for i in range(1, len(self.last_measurement)):
                cv2.line(frame, (self.last_measurement[i][0], self.last_measurement[i][1]),
                         (self.last_measurement[i-1][0], self.last_measurement[i-1][1]), (0, 255, 0), 1)

                cv2.line(frame, (self.last_prediction[i][0], self.last_prediction[i][1]),
                         (self.last_prediction[i-1][0], self.last_prediction[i-1][1]), (255, 255, 0), 1)


    kalman = cv2.KalmanFilter(4, 2, 1)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03


