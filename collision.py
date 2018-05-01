import ballInfo
import math
from collections import deque

height = width = 0
pw = ph = 0


joinEdge =[]
d_p1_p2 = deque(maxlen=3)
d_p1_r = deque(maxlen=3)

def init(w, h, p_w, p_h):
    global width, height, pw, ph
    width = w
    height = h
    pw = p_w
    ph = p_h


def withEdge(color, last_prediction):
    if len(last_prediction) > 2:

        pre_dy = ballInfo.queue[color][2][1] - ballInfo.queue[color][1][1]
        dy = ballInfo.queue[color][1][1] - ballInfo.queue[color][0][1]
        directionY = pre_dy * dy
        if pre_dy == dy == 0:
            directionY = 1

        if directionY <= 0 and last_prediction[0][1] >= height - ph:
            print('B')
            if 'B' not in joinEdge:
                joinEdge.append('B')
                ballInfo.join.append('Edge')
        elif 'B' in joinEdge and last_prediction[0][0] < height - ph:
            joinEdge.remove('B')

        if directionY <= 0 and last_prediction[0][1] <= ph:
            print('U')
            if 'U' not in joinEdge:
                joinEdge.append('U')
                ballInfo.join.append('Edge')
        elif 'U' in joinEdge and last_prediction[0][1] > ph:
            joinEdge.remove('U')

        pre_dx = ballInfo.queue[color][2][0] - ballInfo.queue[color][1][0]
        dx = ballInfo.queue[color][1][0] - ballInfo.queue[color][0][0]
        directionX = pre_dx * dx
        if pre_dx == dx == 0:
            directionX = 1

        if directionX <= 0 and last_prediction[0][0] <= pw:
            print('L')
            if 'L' not in joinEdge:
                joinEdge.append('L')
                ballInfo.join.append('Edge')
        elif 'L' in joinEdge and last_prediction[0][1] > pw:
            joinEdge.remove('L')

        if directionX <= 0 and last_prediction[0][0] >= width - pw:
            print('R')
            if 'R' not in joinEdge:
                joinEdge.append('R')
                ballInfo.join.append('Edge')
        elif 'R' in joinEdge and last_prediction[0][1] < width - pw:
            joinEdge.remove('R')


def withBall(p1, p2, r, success, predict):
    kkk = getDistance(predict, ballInfo.queue[p1][0])
    #print(kkk)
    upper_p1_p2 = (ballInfo.radius[p1] + ballInfo.radius[p2]) * 1.2
    upper_p1_r = (ballInfo.radius[p1] + ballInfo.radius[r]) * 1.2

    p1_p2 = getDistance(ballInfo.queue[p1][0], ballInfo.queue[p2][0])
    d_p1_p2.appendleft(p1_p2)

    while len(d_p1_p2) < 3:
        d_p1_p2.appendleft(p1_p2)

    p1_r = getDistance(ballInfo.queue[p1][0], ballInfo.queue[r][0])
    d_p1_r.appendleft(p1_r)

    while len(d_p1_r) < 3:
        d_p1_r.appendleft(p1_r)

    a1 = (d_p1_p2[1] - d_p1_p2[0]) * (d_p1_p2[2] - d_p1_p2[1])
    a2 = (d_p1_r[1] - d_p1_r[0]) * (d_p1_r[2] - d_p1_r[1])
    '''
    if a1<0:
        print(kkk)
        print("p2:",d_p1_p2)
    if a2 < 0:
        print(kkk)
        print("r:",d_p1_r)
    '''

    '''
    print('-'*20)
    print(d_p1_p2[1] , upper_p1_p2*2.0/1.2)
    print(a1)
    print(kkk, 3*ballInfo.radius[p1])
    '''

    if p2 not in ballInfo.check and ((p1_p2 < upper_p1_p2)
                                     or (d_p1_p2[1] < upper_p1_p2*2.0/1.2 and a1 < 0 and kkk > 2*ballInfo.radius[p1]) ):
        #print(ballInfo.move[p1])
        ballInfo.join.append(p2)
        ballInfo.check.append(p2)
        if not success and r in ballInfo.join:
            if ballInfo.join.count('Edge') >= 3:
                success = True
                ballInfo.join = ["GET SCORE!"]

    elif p2 in ballInfo.check and p1_p2 >= upper_p1_p2:
        ballInfo.check.remove(p2)

    p1_r = getDistance(ballInfo.queue[p1][0], ballInfo.queue[r][0])
    if r not in ballInfo.check and ((p1_r < upper_p1_r)
                                    or (d_p1_r[1] < upper_p1_r * 2.0 / 1.2 and a2 < 0 and kkk > 2 * ballInfo.radius[p1]) ):
        ballInfo.join.append(r)
        ballInfo.check.append(r)
        if not success and p2 in ballInfo.join:
            if ballInfo.join.count('Edge') >= 3:
                success = True
                ballInfo.join = ["GET SCORE!"]

    elif r in ballInfo.check and p1_r >= upper_p1_r:
        ballInfo.check.remove(r)

    return success

def getDistance(o1, o2):
    return math.sqrt((o1[0]-o2[0])**2 + (o1[1]-o2[1])**2)


def check1():
    # 공 간의 거리가 최소가 되는 지점!

    # 예상 위치(0)와 실제 위치의 거리가 3*r 이상

    # 벡터값 ..
    pass