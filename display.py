import cv2
import ballInfo

pw = 18
ph = 16

start_time = 0
pre_frame_time = 0

BGR = {"red":(0,0,255),
        "yellow":(0,255,255),
        "white":(255,255,255),
        "green":(0,255,0),
        "blue":(255,0,0) }


def setStartTime(time):
    global start_time, pre_frame_time
    start_time = time
    pre_frame_time = start_time


def drawLines(img):
    cv2.line(img, (pw, ph), (img.shape[1]-pw, ph), (0,0,255), 1)
    cv2.line(img, (img.shape[1]-pw, ph), (img.shape[1]-pw, img.shape[0]-ph), (0, 0, 255), 1)
    cv2.line(img, (img.shape[1]-pw, img.shape[0]-ph), (pw, img.shape[0]-ph), (0, 0, 255), 1)
    cv2.line(img, (pw, img.shape[0]-ph), (pw, ph), (0, 0, 255), 1)


def displayFPSInfo(img, time, frame_count):
    global start_time, pre_frame_time
    time_m = "Time : %0.2f" % (time-start_time)
    frame_m = "Frame : %d" % frame_count
    fps_m = "FPS : %0.2f" % (1 / (time - pre_frame_time))
    pre_frame_time = time

    cv2.putText(img, time_m, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(img, frame_m, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(img, fps_m, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


def displayScore(frame, y, w):
    cv2.putText(frame, str(y), (450, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, BGR['yellow'])
    cv2.putText(frame, ':', (510, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
    cv2.putText(frame, str(w), (540, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, BGR['white'])


def displayPath(frame, p1):
    for i in range(1, len(ballInfo.queue[p1])):
        #cv2.circle(frame, (ballInfo.queue[p1][i][0], ballInfo.queue[p1][i][1]), 1, BGR[p1], thickness=1)
        cv2.line(frame, (ballInfo.queue[p1][i][0], ballInfo.queue[p1][i][1]),
                 (ballInfo.queue[p1][i - 1][0], ballInfo.queue[p1][i - 1][1]), BGR[p1], 1)


def displayState(frame, p1, state):
    cv2.putText(frame, p1, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BGR[p1])
    cv2.putText(frame, str(ballInfo.join), (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR[p1])
    cv2.putText(frame, state, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR[p1])


def displayMove(frame):
    if ballInfo.move['yellow'][0] == -1:
        pass
    elif ballInfo.move['yellow'][0] < 1.4:
        cv2.circle(frame, (550, 280), 5, BGR['yellow'], thickness=1)
    else:
        cv2.circle(frame, (550, 280), 3, BGR['yellow'], thickness=3)

    if ballInfo.move['white'][0] == -1:
        pass
    elif ballInfo.move['white'][0] < 1.4:
        cv2.circle(frame, (565, 280), 5, BGR['white'], thickness=1)
    else:
        cv2.circle(frame, (565, 280), 3, BGR['white'], thickness=3)

    if ballInfo.move['red'][0] == -1:
        pass
    elif ballInfo.move['red'][0] < 1.4:
        cv2.circle(frame, (580, 280), 5, BGR['red'], thickness=1)
    else:
        cv2.circle(frame, (580, 280), 3, BGR['red'], thickness=3)


def displayKF(frame, last_measurement, last_prediction):
    if len(last_prediction) >= 2:
        for i in range(1, len(last_prediction)):
            #cv2.line(frame, (last_measurement[i][0], last_measurement[i][1]),
            #         (last_measurement[i - 1][0], last_measurement[i - 1][1]), (0, 255, 0), 1)

            cv2.line(frame, (last_prediction[i][0], last_prediction[i][1]),
                     (last_prediction[i - 1][0], last_prediction[i - 1][1]), (255, 255, 0), 1)