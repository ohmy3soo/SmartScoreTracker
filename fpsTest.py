import cv2
import numpy as np
import time

start_time = time.time()
frame_count = 0;

camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    frame_count += 1
    cur_time = time.time() - start_time
    time_m = "Time : %0.2f" % cur_time
    frame_m = "Frame : %d" % frame_count
    fps_m = "FPS: %0.2f" % (frame_count / cur_time)
    cv2.putText(img, time_m, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(img, frame_m, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(img, fps_m, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


    cv2.imshow("fuck", img)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
        break

camera.release()
cv2.destroyAllWindows()