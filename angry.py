import cv2
from exact_face import exact_face
import os
def angry(gray,frame,x,y,w,h):
    pathh = os.path.dirname(os.path.abspath(__file__))
    backred=cv2.imread(pathh+'/Flame/Fi2.jpg')
    backred=cv2.resize(backred,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
    (mask,(y1,y2,x1,x2))=exact_face(gray,frame, x,y,w,h)
    backred=backred[0:y2-y1,0:x2-x1]
    frame1=frame[y1:y2,x1:x2]
    mask_inv = cv2.bitwise_not(mask)
    img_Mblur = cv2.medianBlur(gray[y1:y2,x1:x2], 7)
    img_edge = cv2.adaptiveThreshold(img_Mblur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 1)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img1_back = cv2.bitwise_and(backred,backred,mask = mask)
    img_edge1 = cv2.bitwise_and(img_edge,img_edge,mask = mask)
    backedg=cv2.bitwise_and(img1_back,img_edge1)
    img_frame = cv2.bitwise_and(frame1,frame1,mask = mask_inv)
    dst = cv2.add(img_frame,backedg)
    frame[y1:y2,x1:x2]=dst
    return frame