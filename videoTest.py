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



videoList = ["/Users/kihunahn/Desktop/videoSrc/1.mp4",
                "/Users/kihunahn/Desktop/videoSrc/2.mp4",
                "/Users/kihunahn/Desktop/videoSrc/3.mp4",
                "/Users/kihunahn/Desktop/videoSrc/4.mp4",
                "/Users/kihunahn/Desktop/videoSrc/hard1.mp4",
                "/Users/kihunahn/Desktop/videoSrc/hard2.mp4",
                "/Users/kihunahn/Desktop/videoSrc/hard3.mp4", ###
                "/Users/kihunahn/Desktop/videoSrc/hard4.mp4" ]

videoName = videoList[0]

camera = cv2.VideoCapture(videoName)
ret, img = camera.read()
img = imutils.resize(img, width=600)
billiardFunction.setMatrix(img)

last_prediction = deque()
last_measurement = deque()


def isStop(input):
    for d in input:
        if d != 0 :
            return False
    return True


def onChange(x):
    print('k')
    pass


def KF(position):
    global current_measurement, measurements, last_measurement, current_prediction, last_prediction
    collision.withEdge(last_prediction)

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



p1 = 'yellow'
p2 = 'white'
r = 'red'
success = False

score = {'yellow': 0, 'white': 0}

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Be sure to use the lower case
#out = cv2.VideoWriter('/Users/kihunahn/Desktop/storage/' + str(time.time())+'.avi', fourcc, 30.0, (612, 306))


start_time = time.time()
display.setStartTime(start_time)

frame_count = 0;

frame = billiardFunction.getWarp(img)
width = frame.shape[1]
height = frame.shape[0]

collision.init(width, height, pw, ph)


ballInfo.findBall(r, frame)
ballInfo.findBall(p1, frame)
ballInfo.findBall(p2, frame)

state = 'End'

'''
cv2.namedWindow('12')
cv2.createTrackbar('test', '12', 1, 10, onChange)
'''
cv2.namedWindow('w')
cv2.createTrackbar("B", 'w', 0, 1, onChange)

while camera.isOpened():
    ret, img = camera.read()
    img = imutils.resize(img, width=600)
    #btn1 = cv2.getTrackbarPos('Switch', '12')
    #print(btn1)


    frame_count += 1

    frame = billiardFunction.getWarp(img)

    kernel = np.ones((3,3), np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    ballInfo.traceBall(r, frame)
    ballInfo.traceBall(p1, frame)
    ballInfo.traceBall(p2, frame)

    cX, cY = ballInfo.queue[p1][0]
    KF(ballInfo.queue[p1][0])
    success = collision.withBall(p1, p2, r, success)

    if isStop(ballInfo.move[p1]) and isStop(ballInfo.move[p2]) and isStop(ballInfo.move[r]):
        if state == 'Start':
            if not success:
                print('change')
                temp = p1
                p1 = p2
                p2 = temp
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
            #out.release()
            #fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Be sure to use the lower case
            #out = cv2.VideoWriter('/Users/kihunahn/Desktop/storage/' + str(time.time()) + '.avi', fourcc, 30.0, (612, 306))

    if ballInfo.move[p1][0] > 1.5:
        state = 'Start'

    display.displayState(frame, p1, state)
    display.displayMove(frame)
    display.displayPath(frame, p1)
    display.displayFPSInfo(frame, time.time(), frame_count)
    display.displayScore(frame, score['yellow'], score['white'])

    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', 0, 0)

    key = cv2.waitKey(1)
    #out.write(frame)
    #print(key)
    if key & 0xFF == ord('q'):
        break

#out.release()
camera.release()
cv2.destroyAllWindows()