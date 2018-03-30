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


join = []

camera = cv2.VideoCapture("/Users/kihunahn/Desktop/videoSrc/1.mp4")
ret, img = camera.read()
img = imutils.resize(img, width=600)

billiardFunction.setMatrix(img)
s = []


def getDistance(o1, o2):
    return math.sqrt((o1[0]-o2[0])**2 + (o1[1]-o2[1])**2)



measurements = []
predictions = []
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


#pre_dx = 0
#dx = 0
def KF(frame, x,y):
    global pre_dx, dx
    global current_measurement, measurements, last_measurement, current_prediction, last_prediction
    #print("KF: ", x, y)

    '''
    if len(ballInfo.yellowQ) > 2:
        pre_dx = ballInfo.yellowQ[2][0] - ballInfo.yellowQ[1][0]
        dx = ballInfo.yellowQ[1][0] - ballInfo.yellowQ[0][0]

    direction = pre_dx * dx
    if direction < 0:
        print(direction)

    '''

    # 마우스 이벤트를 통해 새로 들어온 정보로 (x,y) 생성
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])

    # 방금 들어온 자료를 correct, predict
    kalman.correct(current_measurement)
    current_prediction = kalman.predict()

    # last자료를 업데이트
    last_prediction.appendleft(current_prediction)
    last_measurement.appendleft(current_measurement)
    # 점 찍기 위해 위치성분 가져오기

    '''
    lmx, lmy = last_measurement[0], last_measurement[1]
    cmx, cmy = current_measurement[0], current_measurement[1]
    lpx, lpy = last_prediction[0], last_prediction[1]
    cpx, cpy = current_prediction[0], current_prediction[1]
    '''

    # 점 이어서 선 그리기
    if ballInfo.yellowQ[0][0] < width - pw and current_prediction[0] > width - pw:
        print("www")
        s.append("EdgeR")

    if len(last_prediction) >= 2:
        #print(len(last_prediction))
        for i in range(1, len(last_measurement)):
            #print(last_prediction[0])
            #print(last_prediction[1])
            #cv2.line(frame, (lmx, lmy), (cmx, cmy), (0,100,0))
            #cv2.line(frame, (lpx, lpy), (cpx, cpy), (0,0,200))
            #print(last_prediction[0])
            cv2.line(frame, (last_measurement[i][0], last_measurement[i][1]),
                     (last_measurement[i-1][0], last_measurement[i-1][1]), (0, 255, 0), 1)

            cv2.line(frame, (last_prediction[i][0], last_prediction[i][1]),
                     (last_prediction[i-1][0], last_prediction[i-1][1]), (255, 255, 0), 1)


kalman = cv2.KalmanFilter(4,2,1)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*2



p1 = 'yellow'
p2 = 'white'
r = 'red'
success = False

while True:

    ret, img = camera.read()
    img = imutils.resize(img, width=600)

    frame = billiardFunction.getWarp(img)
    width = frame.shape[1]
    height = frame.shape[0]

    rV = ballInfo.traceBall('red', frame)
    yV = ballInfo.traceBall('yellow', frame)
    wV = ballInfo.traceBall('white', frame)

    p1_p2 = getDistance(ballInfo.queue[p1][0], ballInfo.queue[p2][0])
    p1_r = getDistance(ballInfo.queue[p1][0], ballInfo.queue[r][0])

    if p2 not in join and p1_p2 <= (ballInfo.radius[p1] + ballInfo.radius[p2]) * 1.15:
        join.append(p2)
        s.append(p2)
        if not success and r in s:
            if s.count('Edge') >= 3:
                print("GET SCORE")
                success = True
            else:
                s = []
                s.append(p2)

    elif p2 in join and p1_p2 > (ballInfo.radius[p1] + ballInfo.radius[p2]) * 1.15:
        #print(p2, "ball is detached")
        join.remove(p2)

    if r not in join and p1_r <= (ballInfo.radius[p1] + ballInfo.radius[r]) * 1.15:
        join.append(r)
        s.append(r)
        if not success and p2 in s:
            if s.count('Edge') >= 3:
                print("GET SCORE")
                success = True

            else:
                s = [];
                s.append(r)

    elif r in join and p1_r > (ballInfo.radius[p1] + ballInfo.radius[r]) * 1.15:
        #print("Red ball is detached")
        join.remove(r)


    cX, cY = ballInfo.queue[p1][0]
    if "EdgeL" not in join and cX <= pw:
        #print('Touch Left Edge')
        s.append("Edge")
        join.append("EdgeL")
    if "EdgeL" in join and cX > pw:
        join.remove("EdgeL")

    if "EdgeR" not in join and cX >= width - pw:
        #print('Touch Right Edge')
        s.append("Edge")
        join.append("EdgeR")
    if "EdgeR" in join and cX < width - pw:
        join.remove("EdgeR")

    if "EdgeU" not in join and cY <= ph:
        #print('Touch Upper Edge')
        s.append("Edge")
        join.append("EdgeU")
    if "EdgeU" in join and cY > ph:
        join.remove("EdgeU")

    if "EdgeB" not in join and cY >= height - ph:
        #print('Touch Bottom Edge')
        s.append("Edge")
        join.append("EdgeB")
    if "EdgeB" in join and cY < height - ph:
        join.remove("EdgeB")

    if yV == 0 and rV == 0 and wV == 0:
        if not success and s != []:
            temp = p1
            p1 = p2
            p2 = temp
        success = False
        #print('stop')
        s.clear()

    '''
    if len(ballInfo.queue['white']) >= 2:
        for i in range(1, len(ballInfo.queue['white'])):
            cv2.line(frame, (ballInfo.queue['white'][i][0], ballInfo.queue['white'][i][1]),
                    (ballInfo.queue['white'][i-1][0], ballInfo.queue['white'][i-1][1]), (0, 0, 255), 1)
    '''

    e2 = cv2.getTickCount()

    drawLines(frame)


    ##################################################
    #if yX != 0 and yY != 0:
    #KF(frame, cX, cY)
    ##################################################
    cv2.putText(frame, p1, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
    cv2.imshow('frame', frame)
    #out.write(frame)
    print(s)
    #print(join)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
#out.release()
camera.release()
cv2.destroyAllWindows()