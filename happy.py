import cv2
from exact_face import exact_faceH
import os
import numpy as np

def happy(gray, frame, x, y, w, h):
    pathh = os.path.dirname(os.path.abspath(__file__))
    (mask, (y1, y2, x1, x2)) = exact_faceH(gray, frame, x, y, w, h)
    dim = cv2.imread(pathh + '/Flame/dim2.jpg')
    dim = cv2.resize(dim, (y2 - y1, x2 - x1), interpolation=cv2.INTER_AREA)
    # gray1=gray.copy()[y:y+h,x:x+w]
    dimg = cv2.cvtColor(dim, cv2.COLOR_BGR2GRAY)
    frame1 = frame.copy()[y1:y2, x1:x2]
    frame2 = frame.copy()[y1:y2, x1:x2]
    (thresh, thresh) = cv2.threshold(dimg, 130, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    f, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    # contarea=[cv2.contourArea(cnt)  for cnt in contours]
    # inx=np.argsort(contarea)
    # contours=np.array(contours)[inx]
    colors = [(98, 189, 24), (100, 149, 237), (250, 250, 60), (22, 221, 53), (143, 22, 178), (210, 16, 52)]
    for cnt in contours[0:-1]:
        contour = cv2.approxPolyDP(cnt, 4, True)
        R = np.random.choice(len(colors))
        cv2.drawContours(frame1, [contour], -1, colors[R], -1)

    mask_inv = cv2.bitwise_not(mask)
    img_frame1 = cv2.bitwise_and(frame1, frame1, mask=mask)
    img_frame2 = cv2.bitwise_and(frame2, frame2, mask=mask_inv)
    frame[y1:y2, x1:x2] = cv2.add(img_frame1, img_frame2)

    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([45, 45, 45], dtype="uint8")
    # greenimg = np.zeros(frame.shape, np.uint8)
    # greenimg[:] = (0, 255, 0)
    greenimg = cv2.imread(pathh + '/Flame/hair.jpg')
    (gx, gy) = frame.shape[0:2]
    greenimg = cv2.resize(greenimg, (gy, gx), interpolation=cv2.INTER_AREA)
    mask = cv2.inRange(frame, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    frame_img = cv2.bitwise_and(frame, frame, mask=mask_inv)
    green_img = cv2.bitwise_and(greenimg, greenimg, mask=mask)
    frame = cv2.add(green_img, frame_img)

    '''
    mouth=cv2.imread(pathh+'\Flame\Mo.jpg')
    head=frame[y+h*0.8:y+h,x+w*0.2:x+w*0.8]
    ra,ca,dpt = head.shape
    mouth = cv2.resize(mouth,(ca,ra),interpolation = cv2.INTER_AREA)
    mouthg = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY) 
    mouthg = cv2.adaptiveThreshold(mouthg, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 4)
    ret, mask = cv2.threshold(mouthg, 10, 255, cv2.THRESH_BINARY)
    #mouth=cv2.cvtColor(mouth, cv2.COLOR_GRAY2RGB)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(head,head,mask = mask)
    img2_fg = cv2.bitwise_and(mouth,mouth,mask = mask_inv)
    head = cv2.add(img1_bg,img2_fg)

    #frame=halffacecarton(gray,frame,(x,y,w,h)) 
    frame[y+h*0.8:y+h,x+w*0.2:x+w*0.8]=head'''

    # mask = np.zeros(gray.shape[:2], np.uint8)
    # cv2.drawContours(mask, [contour], -1, 255, -1)

    return frame
