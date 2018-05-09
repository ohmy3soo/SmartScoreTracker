import cv2
import numpy as np
import imutils
import billiardFunction
import ballInfo
from collections import deque
import time
import display
import collision

pw = 18
ph = 16

BGRcolor = {"red":(0,0,255),
            "yellow":(0,255,255),
            "white":(255,255,255),
            "green":(0,255,0),
            "blue":(255,0,0) }

videoPath = "/Users/kihunahn/Desktop/videoSrc/"
fps = ["fps30/", "fps60/"]
videoList = ["1.avi", "2.avi", "3.avi", "4.avi", "hard1.avi", "hard2.avi", "hard3.avi",
             "final1.avi", "final2.avi", "final3.avi"]###

videoName = videoPath + fps[0] + videoList[-3]
camera = cv2.VideoCapture(videoName)
frame_count = 0;
def onChange(x):
    pass

ret, img = camera.read()
img = imutils.resize(img, width=600)
billiardFunction.setMatrix(img)
frame = billiardFunction.getWarp(img)
width = frame.shape[1]
height = frame.shape[0]

def isStop(moveList):
    for move in moveList:
        if move != 0:
            return False
    return True





def KF(color, position):
    global current_measurement, measurements, last_measurement, current_prediction, last_prediction, turn
    if color != turn:
        turn = color
        last_prediction.clear()
        last_measurement.clear()

    collision.withEdge(color, last_prediction)
    #print(collision.joinEdge)
    x = position[0]
    y = position[1]

    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])

    kalman.correct(current_measurement)
    current_prediction = kalman.predict()

    last_prediction.appendleft(current_prediction)
    last_measurement.appendleft(current_measurement)

    #display.displayKF(frame, last_measurement, last_prediction)

kalman = cv2.KalmanFilter(4,2,1)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)
last_prediction = deque()
last_measurement = deque()


#t1 = time.localtime()
#outputName = '{}{}{}_{}{}{}'.format(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Be sure to use the lower case
out = cv2.VideoWriter('/Users/kihunahn/Desktop/' + 'final1.avi' , fourcc, 30.0, (612, 306))

p1 = turn = 'yellow'
p2 = 'white'
r = 'red'
state = 'End'
success = False
score = {'yellow': 0, 'white': 0}

display.setStartTime(time.time())
collision.init(width, height, pw, ph)
ballInfo.init(width, height, pw, ph)

ballInfo.findBall(r, frame)
ballInfo.findBall(p1, frame)
ballInfo.findBall(p2, frame)

cv2.namedWindow('frame')
cv2.createTrackbar("Ball", 'frame', False, True, onChange)
cv2.createTrackbar("State", 'frame', True, True, onChange)
cv2.createTrackbar("Move", 'frame', True, True, onChange)
cv2.createTrackbar("Path", 'frame', True, True, onChange)
cv2.createTrackbar("FPS", 'frame', False, True, onChange)
cv2.createTrackbar("Score", 'frame', False, True, onChange)

while camera.isOpened():
    ret, img = camera.read()

    img = imutils.resize(img, width=600)

    frame_count += 1
    # print(frame_count)

    displayBall = cv2.getTrackbarPos('Ball', 'frame')
    displayState = cv2.getTrackbarPos('State', 'frame')
    displayMove = cv2.getTrackbarPos('Move', 'frame')
    displayPath = cv2.getTrackbarPos('Path', 'frame')
    displayFPS = cv2.getTrackbarPos('FPS', 'frame')
    displayScore = cv2.getTrackbarPos('Score', 'frame')

    frame = billiardFunction.getWarp(img)

    kernel = np.ones((3,3), np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    ballInfo.traceBall(r, frame, display=displayBall)
    ballInfo.traceBall(p1, frame, display=displayBall)
    ballInfo.traceBall(p2, frame, display=displayBall)

    KF(p1, ballInfo.queue[p1][0])
    if len(last_prediction) > 2:
        success = collision.withBall(p1, p2, success, last_prediction[1], remain=r) \
                  or collision.withBall(p1, r, success, last_prediction[1], remain=p2)

    if isStop(ballInfo.move[p1]) and isStop(ballInfo.move[p2]) and isStop(ballInfo.move[r]):

        if state == 'Start':
            #print(success)
            if not success:
                print('change')
                temp = p1
                p1 = p2
                p2 = temp
                collision.d_p1_p2.clear()
                collision.d_p1_r.clear()
            else:
                score[p1] += 1
            state = 'End'
            ballInfo.join.clear()
            print('stop')
            success = False
            ballInfo.join = []

            temp_0 = ballInfo.queue[p1][0]
            temp_1 = ballInfo.queue[p1][1]
            ballInfo.queue[p1].clear()
            ballInfo.queue[p1].appendleft(temp_1)
            ballInfo.queue[p1].appendleft(temp_0)
            out.release()
            
            t1 = time.localtime()
            outputName = '{}{}{}_{}{}{}'.format(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Be sure to use the lower case
            out = cv2.VideoWriter('/Users/kihunahn/Desktop/storage/' + outputName + '.avi', fourcc, 30.0, (612, 306))
            
    if ballInfo.move[p1][0] > 1.5:
        state = 'Start'

    if displayState:
        display.displayState(frame, p1, state)
    if displayMove:
        display.displayMove(frame)
    if displayPath:
        display.displayPath(frame, p1)
    if displayFPS:
        display.displayFPSInfo(frame, time.time(), frame_count)
    if displayScore:
        display.displayScore(frame, score['yellow'], score['white'])

    if displayFPS:
        display.displayFPSInfo(frame, time.time(), frame_count)
    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', 0, 0)

    key = cv2.waitKey(1)
    #out.write(frame)
    if key & 0xFF == ord('q'):
        break

#out.release()
camera.release()
cv2.destroyAllWindows()