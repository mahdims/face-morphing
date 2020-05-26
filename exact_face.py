import cv2
import numpy as np

def exact_faceH(gray,frame,x,y,w,h):
    xt = -45
    yt = -20
    frame1 = frame.copy()
    cv2.line(frame1, (x, y), (x+h, y+w), (0, 0, 0), 3)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = gray.copy()[y-yt:y+h+yt,x-xt:x+w+xt]

    (thresh, thresh) = cv2.threshold(gray1, 130, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    f, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
    contarea = [cv2.contourArea(cnt) for cnt in contours]
    cnt = contarea.index(max(contarea))
    contour = cv2.approxPolyDP(contours[cnt], 3, True)
    mask = np.zeros(gray1.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return [mask, (y-yt, y+h+yt, x-xt, x+w+xt)]


def exact_face(gray, frame, x, y, w, h):
    xt = -45
    yt = -20
    gray = gray[y-yt:y+h+yt, x-xt:x+w+xt]
    (thresh, thresh) = cv2.threshold(gray, 130, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    f, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
    contarea = [cv2.contourArea(cnt) for cnt in contours]
    cnt = contarea.index(max(contarea))
    contour = cv2.approxPolyDP(contours[cnt], 3, True)
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return [mask, (y-yt, y+h+yt, x-xt, x+w+xt)]
