import cv2
import numpy as np
import imutils
import billiardFunction
import math
import ballInfo
from collections import deque
import time


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


stopCounter_p1 = 10
stopCounter_p2 = 10
stopCounter_r = 10
stopBuffer = 5


def getDistance(o1, o2):
    return math.sqrt((o1[0]-o2[0])**2 + (o1[1]-o2[1])**2)


'''
last_measurement = current_measurement = np.array((2,1), np.float32)
last_prediction = current_prediction = np.zeros((2,1), np.float32)
'''
last_prediction = deque()
last_measurement = deque()
#current_measurement = np.array((2,1), np.float32)
#current_prediction = np.zeros((2,1), np.float32)

#last_measurement.appendleft(current_measurement)
#last_prediction.appendleft(current_prediction)

IP1 = 0
b1 = 0
b2 = 0


def KF(frame, x,y, draw=True):
    global current_measurement, measurements, last_measurement, current_prediction, last_prediction , IP1
    global b1, b2

    if len(last_prediction) > 2:
            pre_dy = ballInfo.yellowQ[2][1] - ballInfo.yellowQ[1][1]
            dy = ballInfo.yellowQ[1][1] - ballInfo.yellowQ[0][1]

            directionY = pre_dy * dy
            if pre_dy == dy == 0:
                directionY = 1
            if directionY <= 0 and last_prediction[0][1] > height - ph:
                if 'B' not in join:
                    join.append('B')
                    s.append('Edge')
                    print('B')
            elif 'B' in join and last_prediction[0][1] <= height - ph:
                join.remove('B')

            if directionY <= 0 and last_prediction[0][1] <= ph:
                if 'U' not in join:
                    join.append('U')
                    s.append('Edge')
                    print('U')
            elif 'U' in join:
                join.remove('U')

            pre_dx = ballInfo.yellowQ[2][0] - ballInfo.yellowQ[1][0]
            dx = ballInfo.yellowQ[1][0] - ballInfo.yellowQ[0][0]
            directionX = pre_dx * dx
            if pre_dx == dx == 0:
                directionX = 1
            if directionX <= 0 and last_prediction[0][0] <= pw:
                if 'L' not in join:
                    join.append('L')
                    s.append('Edge')
                    print('L')
            elif 'L' in join:
                join.remove('L')

            if directionX <= 0 and last_prediction[0][0] >= width - pw:
                if 'R' not in join:
                    join.append('R')
                    s.append('Edge')
                    print('R')
            elif 'R' in join:
                join.remove('R')


            '''
            x1 = x - last_prediction[0][0]
            y1 = y - last_prediction[0][1]

            x2 = ballInfo.queue[p2][0][0] - ballInfo.queue[p2][1][0]
            y2 = ballInfo.queue[p2][0][1] - ballInfo.queue[p2][1][1]

            IP1 = (x1 * x2 + y1 * y2) /( (((x1 ** 2) + (y1 ** 2)) ** 0.5) * (((x2 ** 2) + (y2 ** 2)) ** 0.5))
            b1 = (x1, y1)
            #if getDistance( (0,0), (x1,y1)) > 1.5 :
            #    print(b1)
            b2 = (x2, y2)
            #print(x1, y1)
            #if not np.isnan(IP1):
            #    print(IP1)
            #if dx != 0:
            #    print(dy/dx)
            '''
            preD = getDistance((x,y), last_prediction[0])
            #if preD > 3:
            #    print(preD)


    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])

    kalman.correct(current_measurement)
    current_prediction = kalman.predict()



    last_prediction.appendleft(current_prediction)
    last_measurement.appendleft(current_measurement)
    '''
    if len(last_prediction) >= 2:
        for i in range(1, len(last_measurement)):
            cv2.line(frame, (last_measurement[i][0], last_measurement[i][1]),
                     (last_measurement[i-1][0], last_measurement[i-1][1]), (0, 255, 0), 1)

            cv2.line(frame, (last_prediction[i][0], last_prediction[i][1]),
                     (last_prediction[i-1][0], last_prediction[i-1][1]), (255, 255, 0), 1)
    '''

kalman = cv2.KalmanFilter(4,2,1)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)



p1 = 'yellow'
p2 = 'white'
r = 'red'
success = False

start = False
end = True

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Be sure to use the lower case
#out = cv2.VideoWriter('/Users/kihunahn/Desktop/storage/' + str(time.time())+'.avi', fourcc, 30.0, (612, 306))

start_time = time.time()
frame_count = 0;
c1 = 0
stackX = 0
stackY = 0
pre = 1

d1 = 0
d2 = 0

frame = billiardFunction.getWarp(img)
width = frame.shape[1]
height = frame.shape[0]
ballInfo.findBall(r, frame)
ballInfo.findBall(p1, frame)
ballInfo.findBall(p2, frame)

while True:
    ret, img = camera.read()
    img = imutils.resize(img, width=600)

    frame_count+=1

    frame = billiardFunction.getWarp(img)
    #cv2.imshow("before", frame)
    kernel = np.ones((3,3), np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("after", frame)
    width = frame.shape[1]
    height = frame.shape[0]

    rV = ballInfo.traceBall(r, frame)
    p1V = ballInfo.traceBall(p1, frame)
    p2V = ballInfo.traceBall(p2, frame)


    p1_p2 = getDistance(ballInfo.queue[p1][0], ballInfo.queue[p2][0])
    p1_r = getDistance(ballInfo.queue[p1][0], ballInfo.queue[r][0])


    cX, cY = ballInfo.queue[p1][0]
    KF(frame, cX, cY, draw=False)

    if stopCounter_p1 > stopBuffer and stopCounter_p2 > stopBuffer and stopCounter_r > stopBuffer:
        if start:
            if not success:
                temp = p1
                p1 = p2
                p2 = temp
            end = True
            start = False
            join.clear()
            print('stop')
            success = False
            s = []
            #out.release()
            #fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Be sure to use the lower case
            #out = cv2.VideoWriter('/Users/kihunahn/Desktop/storage/' + str(time.time()) + '.avi', fourcc, 30.0, (612, 306))

    if p1V > 1.5:
        print(p1V)
        start = True
        end = False


    tempR_p1_p2 = (ballInfo.radius[p1] + ballInfo.radius[p2]) * 1.4
    #IP1 = (last_prediction[0][0] - last_measurement[0][0]) * (last_prediction[0][0] - last_measurement[0][0]) \
    #      + (ballInfo.queue[p2][1][0] - ballInfo.queue[p2][0][0]) * (ballInfo.queue[p2][1][1] - ballInfo.queue[p2][0][1])
    #print("WHTIE: ", IP1)

    if p2 not in join and p1_p2 <= (ballInfo.radius[p1] + ballInfo.radius[p2]) * 1.05 and p2V != 0:
        join.append(p2)
        s.append(p2)
        #print("IP1 X")
        if not success and r in s:
            if s.count('Edge') >= 3:
                print("GET SCORE")
                success = True
                s = "GET SCORE!"
            else:
                s = []
                s.append(p2)

    elif p2 not in join and tempR_p1_p2 >= getDistance(ballInfo.queue[p1][0], ballInfo.queue[p2][0]):# and IP1 > 0:
        #print("IP1!!")

        join.append(p2)
        s.append(p2)
        if not success and r in s:
            if s.count('Edge') >= 3:
                print("GET SCORE")
                success = True
                s = ["GET SCORE!"]
            else:
                s = []
                s.append(p2)

    elif p2 in join and p1_p2 > (ballInfo.radius[p1] + ballInfo.radius[p2]) * 1.5:
        print(p2, "ball is detached")
        join.remove(p2)


    tempR_p1_r = (ballInfo.radius[p1] + ballInfo.radius[r]) * 1.4
    IP2 = (last_prediction[0][0] - last_measurement[0][0]) * (last_prediction[0][0] - last_measurement[0][0]) \
          + (ballInfo.queue[r][1][0] - ballInfo.queue[r][0][0]) * (ballInfo.queue[r][1][1] - ballInfo.queue[r][0][1])
    #print("RED: ", IP2)
    if r not in join and p1_r <= (ballInfo.radius[p1] + ballInfo.radius[r]) * 1.05 and rV >1.5:
        join.append(r)
        s.append(r)
        print('IP2 X')
        if not success and p2 in s:
            if s.count('Edge') >= 3:
                print("GET SCORE")
                success = True

            else:
                s = [];
                s.append(r)
    elif r not in join and tempR_p1_r >= getDistance(ballInfo.queue[p1][0], ballInfo.queue[r][0]) and rV != 0:# and IP2 > 0:
        print("IP2!!")
        join.append(r)
        s.append(r)
        if not success and p2 in s:
            if s.count('Edge') >= 3:
                print("GET SCORE")
                success = True
                s = ["GET SCORE!"]
            else:
                s = []
                s.append(r)
    elif r in join and p1_r > (ballInfo.radius[p1] + ballInfo.radius[r]) * 1.5:
        print("Red ball is detached")
        join.remove(r)


    #drawLines(frame)
    #print('IP1: ', IP1)

    cv2.line(img, (pw, ph), (img.shape[1] - pw, ph), (0, 0, 255), 1)
    cv2.line(img, (pw, ph), (img.shape[1] - pw, ph), (0, 0, 255), 1)

    cv2.putText(frame, p1, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BGRcolor[p1])
    cv2.putText(frame, str(s), (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGRcolor[p1])

    if start:
        cv2.putText(frame, 'Start', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGRcolor[p1])
    if end:
        cv2.putText(frame, 'End', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGRcolor[p1])

    if p1V > 0:
        stopCounter_p1 = 0
    else:
        stopCounter_p1 += 1

    if p2V > 0:
        stopCounter_p2 = 0
    else:
        stopCounter_p2 += 1

    if rV > 0:
        stopCounter_r = 0
    else:
        stopCounter_r += 1


    if stopCounter_p1 > stopBuffer:
        cv2.circle(frame, (550, 280), 5, BGRcolor[p1], thickness=1)
    else:
        cv2.circle(frame, (550, 280), 3, BGRcolor[p1], thickness=3)
    if stopCounter_p2 > stopBuffer:
        cv2.circle(frame, (565, 280), 5, BGRcolor[p2], thickness=1)
    else:
        cv2.circle(frame, (565, 280), 3, BGRcolor[p2], thickness=3)
    if stopCounter_r > stopBuffer:
        cv2.circle(frame, (580, 280), 5, BGRcolor[r], thickness=1)
    else:
        cv2.circle(frame, (580, 280), 3, BGRcolor[r], thickness=3)


    #d1_pre = d1
    #d2_pre = d2
    #d1 = getDistance(ballInfo.queue[p1][0], ballInfo.queue[p2][0])
    #d2 = getDistance(ballInfo.queue[p1][0], ballInfo.queue[r][0])

    #print(abs(d1_pre - d1) , abs(d2_pre - d2))
    #print(p1V, p2V, rV)
    
    c2 = time.time()
    cur_time = time.time() - start_time
    time_m = "Time : %0.2f" % cur_time
    frame_m = "Frame : %d" % frame_count
    fps_m = "FPS : %0.2f" % (1/(c2-c1))
    c1 = c2
    cv2.putText(frame, time_m, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(frame, frame_m, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(frame, fps_m, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    #print(ballInfo.check)
    ballInfo.check = []
    cv2.imshow('frame', frame)
    #out.write(frame)
    #print(s)
    #if join != []:
    #    print(join)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break
# Release everything if job is finished
#out.release()
camera.release()
cv2.destroyAllWindows()