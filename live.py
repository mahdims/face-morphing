# -*- coding: cp1254 -*-
# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import numpy as np
import scipy.interpolate as scint
import os
import copy
from itertools import product
pathh=os.path.dirname(os.path.abspath(__file__))


def softmix(A,B):
    ra,ca,dpt = A.shape
    rb,cb,dpt=B.shape 
    #rrate=float(ra)/float(rb)
    #crate=float(ca)/float(cb)
    #B = cv2.resize(B,None,fx=rrate,fy=crate,interpolation = cv2.INTER_AREA)
    B = cv2.resize(B,(ca,ra),interpolation = cv2.INTER_AREA)
    #cv2.imshow('frame2',B)
    level=3
    GA = A.copy()
    gpA = [GA]
    GB = B.copy()
    gpB = [GB]
    for i in xrange(level):
        # generate Gaussian pyramid for A
        #size=np.array(GA.shape[0:2],dtype=int)/2+1
        GA = cv2.pyrDown(GA)#, dstsize=tuple(size))
        gpA.append(GA)
        # generate Gaussian pyramid for B
        GB = cv2.pyrDown(GB)#,dstsize=tuple(size))
        gpB.append(GB)

    LA = gpA[level]
    LB = gpB[level]
    LS = []
    # Now add left and right halves of images in each level
    rows,cols,dpt = LA.shape    
    ls = np.concatenate((LA[:,0:cols/8], LB[:,cols/8:7*cols/8],LA[:,7*cols/8:cols]), axis=1)
    #ls=np.concatenate((LA[0:rows/8,:],ls[rows/8:7*rows/8,:],LA[7*rows/8:rows,:]), axis=0)
    ls=np.concatenate((ls[0:7*rows/8,:],LA[7*rows/8:rows,:]), axis=0)
    
    LS.append(ls)
    for i in xrange(level,0,-1):
        size=np.array(gpA[i-1].shape[0:2],dtype=int)
        # generate Laplacian Pyramid for A
        GE = cv2.pyrUp(gpA[i],dstsize=(size[1],size[0]))
        LA = cv2.subtract(gpA[i-1],GE)
        #lpA.append(LA)

        # generate Laplacian Pyramid for B
        GE = cv2.pyrUp(gpB[i],dstsize=(size[1],size[0]))
        LB = cv2.subtract(gpB[i-1],GE)
        #lpB.append(LB)
        # Now add left and right halves of images in each level
        rows,cols,dpt = LA.shape    
        ls = np.concatenate((LA[:,0:cols/8], LB[:,cols/8:7*cols/8],LA[:,7*cols/8:cols]), axis=1)
        #ls=np.concatenate((LA[0:rows/8,:],ls[rows/8:7*rows/8,:],LA[7*rows/8:rows,:]), axis=0)
        ls=np.concatenate((ls[0:7*rows/8,:],LA[7*rows/8:rows,:]), axis=0)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1,level+1):
        size=np.array(LS[i].shape[0:2],dtype=int)
        ls_ = cv2.pyrUp(ls_,dstsize=(size[1],size[0]))
        ls_ = cv2.add(ls_, LS[i])
    
    return ls_



def create_LUT_8UC1(x, y):
    spl = scint.UnivariateSpline(np.array(x), np.array(y))
    return spl(xrange(256))


def decolorize(frame,(x,y),mood):
    xt=0
    yt=0
    rows,cols,cln=frame.shape
    frame1=frame[y-yt:y+rows,x-xt:x+cols]
    c_r, c_g, c_b = cv2.split(frame1)
    if mood=='sad':
        colmin=[0,63,126,191,255]
        colmax=[0,63,100,200,255]
    else:
        #colmin=[0,150,150,192,250] #HAPPY filter
        #colmax=[0,150,168,192,250]
        colmin=[0,100,128,200,250] # Happy filtr
        colmax=[0,100,128,250,250] 
        
    myLUT=create_LUT_8UC1(colmin,colmax)
    c_r = cv2.LUT(c_r, myLUT).astype(np.uint8)
    c_b = cv2.LUT(c_b, myLUT).astype(np.uint8)
    frame1 = cv2.merge((c_r, c_g, c_b))
    frame[y-yt:y+rows,x-xt:x+cols]=frame1
    return frame
    
def exactface(gray,frame,(x,y,w,h)):
    xt=-45
    yt=-20
    #levels=3
    gray=gray[y-yt:y+h+yt,x-xt:x+w+xt]
    #frame1=frame[y-yt:y+h+yt,x-xt:x+w+xt]
    #ret, thresh = cv2.threshold(gray, 127, 255, 0)
    (thresh, thresh) = cv2.threshold(gray, 130, 255, 
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #                                cv2.THRESH_BINARY,19,12)
    
    #se = np.ones((7,7), dtype='uint8')
    #image_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)
    f, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
    contarea=[cv2.contourArea(cnt)  for cnt in contours]
    cnt=contarea.index(max(contarea))  
    #contour=contours[cnt]                                                                                    
    contour=cv2.approxPolyDP(contours[cnt], 3, True) 
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    #contour=cv2.convexHull(contours[cnt],True,False)
    #cv2.drawContours(frame1, [contour], 0, (0,255,0), 3)
    #cv2.drawContours(frame1, contours, (-1, 2)[levels <= 0], (128,255,255),
     #       3, cv2.LINE_AA, hierarchy, abs(levels) )
    #frame[y-yt:y+h+yt,x-xt:x+w+xt]=frame1
    return [mask,(y-yt,y+h+yt,x-xt,x+w+xt)]
def eyeson(gray1,frame,(x,y,w,h)):
    receyeC=frame[y-h/2:y+h/2,x-w/2:x+w/2]
    gray=gray1.copy()
    cv2.ellipse(gray,(x,y+9),(4*w/6,2*h/8),0,0,360,(0,0,0),-1)
    receyeG=gray[y-h/2:y+h/2,x-w/2:x+w/2,]
    ret, mask = cv2.threshold(receyeG, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    #receyeC=carton(receyeG,receyeC)
    eye = cv2.bitwise_and(receyeC,receyeC,mask = mask_inv)
    
    positions=[[-3*w/2,-2*hl],[-7*w/4,-1*hl/4],[-w,0],[-w,-4*h/3]]#product([-3*wl/2,-wl],[-2*hl,-hl,0])
    for (ysh,xsh) in positions:
        newposition=frame[y-h/2+ysh:y+h/2+ysh,x-w/2+xsh:x+w/2+xsh]
        newposition = cv2.bitwise_and(newposition,newposition,mask = mask)
        dst = cv2.add(newposition,eye)
        frame[y-h/2+ysh:y+h/2+ysh,x-w/2+xsh:x+w/2+xsh]=dst
    #cv2.ellipse(frame,(x,y+9),(4*wl/6,2*hl/8),0,0,360,(255,255,255),-1)
    return frame    


def carton(gray,frame):
    img_Mblur = cv2.medianBlur(gray, 7)
    img_edge = cv2.adaptiveThreshold(img_Mblur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    #img_edge[:,:,0],img_edge[:,:,1],img_edge[:,:,2] =(R,G,B)
    temp=cv2.bitwise_and(frame, img_edge)
    frame=temp
    return frame

def printonimage(img1,img2,(x,y)):
    rows,cols,channels = img2.shape
    roi = img1[y:y+rows,x-cols:x ]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[y:y+rows,x-cols:x ] = dst   
    return img1
def dodge(image, mask):
    return cv2.divide(image, 255-mask, scale=256)
def burn(image, mask):
    return 255-cv2.divide(255-image, 255-mask, scale=256)
    
def angry(gray,frame,(x,y,w,h)):
    backred=cv2.imread(pathh+'\Flame\Blue.jpg')
    backred=cv2.resize(backred,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
    (mask,(y1,y2,x1,x2))=exactface(gray,frame,(x,y,w,h))
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
    
def angry1(gray,frame,(x,y,w,h)):
    mouth=cv2.imread(pathh+'\Flame\mo.png')
    
    head=frame[y+h*0.8:y+h,x+w*0.2:x+w*0.8]
    ra,ca,dpt = head.shape
    mouth = cv2.resize(mouth,(ca,ra),interpolation = cv2.INTER_AREA)
    #mouth=mouth[:,ca:2*ca]
    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY) 
    mouth = cv2.adaptiveThreshold(mouth, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 4)
    ret, mask = cv2.threshold(mouth, 10, 255, cv2.THRESH_BINARY)
    mouth=cv2.cvtColor(mouth, cv2.COLOR_GRAY2RGB)
    mouth[:,:,2] =255
    #cv2.imshow('fff',mouth)
    #img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(head,head,mask = mask)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(mouth,mouth,mask = mask_inv)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
        
    #frame=halffacecarton(gray,frame,(x,y,w,h)) 
    frame[y+h*0.8:y+h,x+w*0.2:x+w*0.8]=dst
    return frame
def happy(gray,frame,(x,y,w,h)):
    frame=decolorize(frame,(0,0),'happy')


    return frame
    
def genlines():
    lines=[]
    for lenght in [9,15,8,4]:
        points=[]
        x,y=0,0
        for _ in range(lenght):
            a=np.random.randint(13)
            b=np.random.randint(20)
            x=x+(-1)**b*a
            c=np.random.randint(20)
            y=y+c
            points.append((x,y))
        lines.append(points)
        
    return lines

def sad1(gray,frame,(x,y,w,h)):
    ###x,y,w,h eyes cordinates
    sea=cv2.imread(pathh+'\Flame\Blue.jpg')
    sea = cv2.resize(sea,(2*h,2*w),interpolation = cv2.INTER_AREA)
    receyeC=frame[y-h:y+h,x-w:x+w]
    cv2.ellipse(gray,(x,y+9),(3*wl/5,3*hl/5),0,0,360,(0,0,0),-1)
    receyeG=gray[y-h:y+h,x-w:x+w]
    #cv2.imshow('frame1', receyeG)
    ret, mask = cv2.threshold(receyeG, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1frame = cv2.bitwise_and(receyeC,receyeC,mask = mask)
    img2sea= cv2.bitwise_and(sea,sea,mask = mask_inv)
    temp=cv2.add(img1frame,img2sea)
    frame[y-h:y+h,x-w:x+w]=temp
    #clines=copy.copy(lines)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame
def sad(gray,frame,(x,y,w,h),(rex,rey),(lex,ley)):
    frame=eyeson(gray,frame,(reye_cent[0],reye_cent[1],wr,hr))
    #clines=copy.copy(lines)
    for points in rlines:
        point=np.int32(points)
        point[:,0]+=rex
        point[:,1]+=rey+8
        cv2.polylines(frame, [point],False,(0,0,0),2)
    for points in llines:
        point=np.int32(points)
        point[:,0]+=lex
        point[:,1]+=ley+8
        cv2.polylines(frame, [point],False,(0,0,0),2)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame=decolorize(frame,(0,y),'sad')
    frame=carton(gray,frame)
    return frame
cap = cv2.VideoCapture(0)
ret = cap.set(3,850) 
ret = cap.set(4,550)
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("C:\haarcascade_frontalface_default.xml")
eye_casc=cv2.CascadeClassifier("C:\haarcascade_eye.xml")
left_eye_casc=cv2.CascadeClassifier("C:\haarcascade_lefteye_2splits.xml")
right_eye_casc=cv2.CascadeClassifier("C:\haarcascade_righteye_2splits.xml")

rlines=genlines()
llines=genlines()
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        #frame=img_blend
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.3,
		minNeighbors=5,
		minSize=(30, 30))
		#flags = cv2.CV_HAAR_SCALE_IMA
   
	# Draw a rectangle around the faces
	try:
       	    for (x, y, w, h) in faces:
       	        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        	   #Detect eyes
        	head=gray[y:y+h,x:x+w]
        	left_eye_region = head[0.2*h:0.5*h,0.1*w:0.5*w]
                left_eye=left_eye_casc.detectMultiScale(left_eye_region)#scaleFactor=1.1, minNeighbors=3) #flags=cv2.CV_HAAR_FIND_BIGGEST_OBJECT)
        	leye_cent=None
        	for (xl, yl, wl, hl) in left_eye:
                     leye_cent=(x+w/10+xl+hl/2,y+h/5+yl+wl/2)
                     
                     break 	
                
                right_eye_region=head[0.2*h:0.5*h,0.5*w:0.9*w]
                right_eye=right_eye_casc.detectMultiScale(right_eye_region)# scaleFactor=1.1, minNeighbors=3)#flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
                reye_cent=None
                for (xr, yr, wr, hr) in right_eye:
                        reye_cent=(x+w/2+xr+wr/2,y+h/5 + yr + hr/2)
                        '''try:
                            frame=gettheeye(gray,frame,(reye_cent[0],reye_cent[1],wr,hr))
                        except:
                            print 'ERORReye' 
                            pass
                        #cv2.ellipse(frame,(x+w/2+xr+wr/2,y+h/5+yr+hr/2),(4*wr/6,2*hr/6),0,0,360,255,2)'''
                        break
                
                
                #frame=halffacecarton(gray,frame,(x,y,w,h))
                #frame=decolorize(frame,(x,y,w,h))
                #(frame,(y1,y2,x1,x2))=exactface(gray,frame,(x,y,w,h))
                #frame=angry(gray,frame,(x,y,w,h))
                #frame=happy(gray,frame,(x,y,w,h))
                #head=frame[y:y+h,x:x+w]
                #frame[y:y+h,x:x+w]=softmix(head,img2)  
                #frame=sad(gray,frame,(x,y,w,h),(reye_cent[0],reye_cent[1]),(leye_cent[0],leye_cent[1]))      
                #frame=sad1(gray,frame,(leye_cent[0],leye_cent[1], wl, hl))
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame=cv2.stylization(frame, sigma_s=30,sigma_r=0.3)
                #(_,frame)=cv2.pencilSketch(frame,sigma_s=10,sigma_r=0.2)
                #frame=cv2.detailEnhance(frame, sigma_s=20,sigma_r=0.2)
                #frame=cv2.edgePreservingFilter(frame,flags=1, sigma_s=60,sigma_r=0.4)
                '''if np.random.rand()>=0.5:
        	           flame=cv2.imread(pathh+'\Flame\l1.png')
       	        else:
        	           flame=cv2.imread(pathh+'\Flame\l2.png')    
       	        flame = cv2.resize(flame,(100,30),interpolation = cv2.INTER_CUBIC)   
       	        frame=printonimage(frame,flame,(x,y))'''
                
                break # first face detected
        except:
            print 'ERORR'
            pass

                
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()