import cv2

pw = 18
ph = 16


def drawLines(img):
    cv2.line(img, (pw, ph), (img.shape[1]-pw, ph), (0,0,255), 1)
    cv2.line(img, (img.shape[1]-pw, ph), (img.shape[1]-pw, img.shape[0]-ph), (0, 0, 255), 1)
    cv2.line(img, (img.shape[1]-pw, img.shape[0]-ph), (pw, img.shape[0]-ph), (0, 0, 255), 1)
    cv2.line(img, (pw, img.shape[0]-ph), (pw, ph), (0, 0, 255), 1)


def displayFPSInfo(img, time, frame, fps):
    pass


def displayScore():
    pass

def displayPath():
    pass

def displayBall():
    pass