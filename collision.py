import ballInfo
import math
from collections import deque

height = width = 0
pw = ph = 0

d_p1_p2 = deque(maxlen=3)
d_p1_r = deque(maxlen=3)

joinEdge = []

temp = {'p2': False, 'r': False}

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

        if directionY <= 0 and pre_dy < 0 and last_prediction[0][1] >= height - ph:
            print('B')
            if 'B' not in joinEdge:
                joinEdge.append('B')
                ballInfo.join.append('Edge(B)')
        elif 'B' in joinEdge and last_prediction[0][0] < height - ph:
            joinEdge.remove('B')

        if directionY <= 0 and pre_dy > 0 and last_prediction[0][1] <= ph:
            print('U')
            if 'U' not in joinEdge:
                joinEdge.append('U')
                ballInfo.join.append('Edge(U)')
        elif 'U' in joinEdge and last_prediction[0][1] > ph:
            joinEdge.remove('U')

        pre_dx = ballInfo.queue[color][2][0] - ballInfo.queue[color][1][0]
        dx = ballInfo.queue[color][1][0] - ballInfo.queue[color][0][0]
        directionX = pre_dx * dx

        if directionX <= 0 and pre_dx > 0 and last_prediction[0][0] <= pw:
            print('L')
            if 'L' not in joinEdge:
                joinEdge.append('L')
                ballInfo.join.append('Edge(L)')
        elif 'L' in joinEdge and last_prediction[0][1] > pw:
            joinEdge.remove('L')

        if directionX <= 0 and pre_dx < 0 and last_prediction[0][0] >= width - pw:
            print('R')
            if 'R' not in joinEdge:
                joinEdge.append('R')
                ballInfo.join.append('Edge(R)')
        elif 'R' in joinEdge and last_prediction[0][1] < width - pw:
            joinEdge.remove('R')


def withBall(b1, b2, success, predict, remain):
    global temp, d_p1_r, d_p1_p2

    diff = getDistance(predict, ballInfo.queue[b1][0])

    limit = (ballInfo.radius[b1] + ballInfo.radius[b2]) * 1.2

    b1_b2 = getDistance(ballInfo.queue[b1][0], ballInfo.queue[b2][0])

    if b2 == 'red':
        update = d_p1_r
        o = 'r'
    else:
        update = d_p1_p2
        o = 'p2'

    update.appendleft(b1_b2)
    while len(update) < 3:
        update.appendleft(b1_b2)

    a = (update[1] - update[0]) * (update[2] - update[1])

    if b1_b2 < limit and not temp[o] and b2 not in ballInfo.check:
        temp[o] = True
        return success

    '''
    if b2 == 'white':
        print(b1_b2, limit)
        print(d_p1_p2)
        print(update[1] , limit * 2.0 / 1.2)
        print(a)
        print(diff, 2.0 * ballInfo.radius[b1])
    '''
    if b2 not in ballInfo.check and ((temp[o] and not isStop(ballInfo.move[b2]))
                                     or (update[1] < limit * 2.0 / 1.2 and a < 0 and diff+1 > ballInfo.radius[b1])):

        ballInfo.join.append(b2)
        ballInfo.check.append(b2)

        if not success and remain in ballInfo.join:
            if ballInfo.join.count('Edge(L)') \
                    + ballInfo.join.count('Edge(R)') \
                    + ballInfo.join.count('Edge(B)') \
                    + ballInfo.join.count('Edge(U)') >= 3:
                success = True
                ballInfo.join = ["GET SCORE!"]
        if temp[o]:
            temp[o] = not temp[o]

    elif b2 in ballInfo.check and b1_b2 >= limit:
        ballInfo.check.remove(b2)

    return success


def getDistance(o1, o2):
    return math.sqrt((o1[0]-o2[0])**2 + (o1[1]-o2[1])**2)



def isStop(moveList):
    for move in moveList:
        if move != 0:
            return False
    return True
