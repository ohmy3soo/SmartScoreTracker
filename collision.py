import ballInfo
from ballInfo import join, check
import math
height = width = 0
pw = ph = 0

def init(w, h, p_w, p_h):
    global width, height, pw, ph
    width = w
    height = h
    pw = p_w
    ph = p_h


def withEdge(last_prediction):
    if len(last_prediction) > 2:

        pre_dy = ballInfo.yellowQ[2][1] - ballInfo.yellowQ[1][1]
        dy = ballInfo.yellowQ[1][1] - ballInfo.yellowQ[0][1]
        directionY = pre_dy * dy
        if pre_dy == dy == 0:
            directionY = 1

        if directionY <= 0 and last_prediction[0][1] > height - ph:
            join.append('Edge')

        if directionY <= 0 and last_prediction[0][1] <= ph:
            join.append('Edge')


        pre_dx = ballInfo.yellowQ[2][0] - ballInfo.yellowQ[1][0]
        dx = ballInfo.yellowQ[1][0] - ballInfo.yellowQ[0][0]
        directionX = pre_dx * dx
        if pre_dx == dx == 0:
            directionX = 1
        if directionX <= 0 and last_prediction[0][0] <= pw:
            join.append('Edge')

        if directionX <= 0 and last_prediction[0][0] >= width - pw:
            join.append('Edge')


def withBall(p1, p2, r, success):
    global join, check

    upper_p1_p2 = (ballInfo.radius[p1] + ballInfo.radius[p2]) * 1.2
    upper_p1_r = (ballInfo.radius[p1] + ballInfo.radius[r]) * 1.2

    p1_p2 = getDistance(ballInfo.queue[p1][0], ballInfo.queue[p2][0])
    if p2 not in check and p1_p2 < upper_p1_p2:
        join.append(p2)
        check.append(p2)
        if not success and r in join:
            if join.count('Edge') >= 3:
                print("GET SCORE")
                success = True
                join = ["GET SCORE!"]

    elif p2 in check and p1_p2 >= upper_p1_p2:
        print(p2, "ball is detached")
        check.remove(p2)

    p1_r = getDistance(ballInfo.queue[p1][0], ballInfo.queue[r][0])
    if r not in check and p1_r < upper_p1_r:
        join.append(r)
        check.append(r)
        if not success and p2 in join:
            if join.count('Edge') >= 3:
                print("GET SCORE")
                success = True
                join = ["GET SCORE!"]
    elif r in check and p1_r >= upper_p1_r:
        print("Red ball is detached")
        check.remove(r)

    return success

def getDistance(o1, o2):
    return math.sqrt((o1[0]-o2[0])**2 + (o1[1]-o2[1])**2)