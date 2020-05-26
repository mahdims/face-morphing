import numpy as np
import cv2
import interface
import wx 
import os
import scipy.interpolate as scint
import copy
from itertools import product

pathh=os.path.dirname(os.path.abspath(__file__))
def printonimage(img1,img2,(x,y,w,h)): 
    rows,cols,channels = img2.shape
    #img2 = cv2.resize(img2,(w,h),interpolation = cv2.INTER_CUBIC)
    roi = img1[y:y+cols,x-rows:x]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    img1[y:y+cols,x-rows:x] = dst   
    return img1
def exactfaceH(gray,frame,(x,y,w,h)):
    xt=-45
    yt=-20
    frame1=frame.copy()
    cv2.line(frame1,(x,y),(x+h,y+w),(0,0,0), 3)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1=gray.copy()[y-yt:y+h+yt,x-xt:x+w+xt]

    (thresh, thresh) = cv2.threshold(gray1, 130, 255, 
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    f, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
    contarea=[cv2.contourArea(cnt)  for cnt in contours]
    cnt=contarea.index(max(contarea))                                                                                   
    contour=cv2.approxPolyDP(contours[cnt], 3, True) 
    mask = np.zeros(gray1.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return [mask,(y-yt,y+h+yt,x-xt,x+w+xt)]    
def exactface(gray,frame,(x,y,w,h)):
    xt=-45
    yt=-20
    gray=gray[y-yt:y+h+yt,x-xt:x+w+xt]
    (thresh, thresh) = cv2.threshold(gray, 130, 255, 
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    f, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
    contarea=[cv2.contourArea(cnt)  for cnt in contours]
    cnt=contarea.index(max(contarea))                                                                                   
    contour=cv2.approxPolyDP(contours[cnt], 3, True) 
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return [mask,(y-yt,y+h+yt,x-xt,x+w+xt)]
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
def eyeson(gray,frame,(xr,yr,wr,hr)):
    receyeC=frame[yr-hr/2:yr+hr/2,xr-wr/2:xr+wr/2]
    gray1=gray.copy()
    cv2.ellipse(gray1,(xr,yr+9),(5*wr/6,2*hr/8),0,0,360,(0,0,0),-1)
    receyeG=gray1[yr-hr/2:yr+hr/2,xr-wr/2:xr+wr/2]
    ret, mask = cv2.threshold(receyeG, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    #receyeC=carton(receyeG,receyeC)
    eye = cv2.bitwise_and(receyeC,receyeC,mask = mask_inv)
    
    positions=[[-3*wr/2,-2*hr],[-7*wr/4,-1*hr/4],[-wr,0],[-wr,-4*hr/3]]#product([-3*wl/2,-wl],[-2*hl,-hl,0])
    for (ysh,xsh) in positions:
        newposition=frame[yr-hr/2+ysh:yr+hr/2+ysh,xr-wr/2+xsh:xr+wr/2+xsh]
        newposition = cv2.bitwise_and(newposition,newposition,mask = mask)
        dst = cv2.add(newposition,eye)
        frame[yr-hr/2+ysh:yr+hr/2+ysh,xr-wr/2+xsh:xr+wr/2+xsh]=dst
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
                
def sad(gray,frame,(x,y,w,h),(rex,rey,wr,hr),(lex,ley)):
    frame=eyeson(gray,frame,(rex,rey,wr,hr))
    #clines=copy.copy(lines)
    for points in rlines:
        point=np.int32(points)
        point[:,0]+=rex
        point[:,1]+=rey+8
        cv2.polylines(frame, [point],False,(0,0,0),3)
    for points in llines:
        point=np.int32(points)
        point[:,0]+=lex
        point[:,1]+=ley+8
        cv2.polylines(frame, [point],False,(0,0,0),3)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame=decolorize(frame,(0,0),'sad')
    frame=carton(gray,frame)
    return frame
        
def angry(gray,frame,(x,y,w,h)):
    backred=cv2.imread(pathh+'\Flame\Fi2.jpg')
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

def happy(gray,frame,(x,y,w,h)):
    (mask,(y1,y2,x1,x2))=exactfaceH(gray,frame,(x,y,w,h))
    dim=cv2.imread(pathh+'\Flame\dim2.jpg')
    dim=cv2.resize(dim,(y2-y1,x2-x1),interpolation = cv2.INTER_AREA)
    #frame=decolorize(frame,(0,0),'happy')
    #gray1=gray.copy()[y:y+h,x:x+w]
    dimg=cv2.cvtColor(dim, cv2.COLOR_BGR2GRAY)
    frame1=frame.copy()[y1:y2,x1:x2]
    frame2=frame.copy()[y1:y2,x1:x2]
    (thresh, thresh) = cv2.threshold(dimg, 130, 255, 
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    f, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
    #contarea=[cv2.contourArea(cnt)  for cnt in contours]
    #inx=np.argsort(contarea)
    #contours=np.array(contours)[inx] 
    colors=[(98,189,24),(100,149,237),(250,250,60),(22,221,53),(143,22,178),(210,16,52)]
    for cnt in contours[0:-1]:                                                                                
        contour=cv2.approxPolyDP(cnt, 4, True)
        R=np.random.choice(len(colors))
        cv2.drawContours(frame1, [contour], -1, colors[R], -1)
        
    
    mask_inv = cv2.bitwise_not(mask)
    img_frame1 = cv2.bitwise_and(frame1,frame1,mask = mask)
    img_frame2 = cv2.bitwise_and(frame2,frame2,mask = mask_inv)
    frame[y1:y2,x1:x2]=cv2.add(img_frame1,img_frame2)
    
    lower = np.array([0,0,0], dtype = "uint8")
    upper = np.array([45,45,45], dtype = "uint8")
    #greenimg = np.zeros(frame.shape, np.uint8)
    #greenimg[:] = (0, 255, 0)
    greenimg=cv2.imread(pathh+'\Flame\hair.jpg')
    (gx,gy)=frame.shape[0:2]
    greenimg=cv2.resize(greenimg,(gy,gx),interpolation = cv2.INTER_AREA)
    mask = cv2.inRange(frame, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    frame_img = cv2.bitwise_and(frame, frame, mask = mask_inv)
    green_img=cv2.bitwise_and(greenimg, greenimg, mask = mask)
    frame=cv2.add(green_img,frame_img)
    
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
    

    #mask = np.zeros(gray.shape[:2], np.uint8)
    #cv2.drawContours(mask, [contour], -1, 255, -1)


    return frame


class GetBut(interface.MyFrame): 
   def __init__(self,parent): 
      interface.MyFrame.__init__(self,parent)  
      
   def initialize(self):
        self.Show(True)	
   def Quit(self, event):
        self.Close() 
                    
   def ShowIcon(self,e): 
        global fcountr
        global lecout
        global recout
        #global x,y 
        #global xl,yl,wl,hl
        #global xr,yr,wr,hr
        #global y
        mood=text[e.GetId()-1]
        #icon = cv2.imread(pathh+'\Icons\icon%d.png' %e.GetId())
        #icon = cv2.resize(icon,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
        cap = cv2.VideoCapture(0)
        ret = cap.set(3,966) 
        ret = cap.set(4,668)
        #face_cascade = cv2.CascadeClassifier()
        #eye_cascade=cv2.CascadeClassifier()
        #eye_cascade.load(pathh+'\haarcascade_eye.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX
        while(True):
            ret, frame = cap.read()
            #if frame.empty(): break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.3,
		minNeighbors=5,
		minSize=(30, 30))
		#flags = cv2.CV_HAAR_SCALE_IMAGE)
	   ## get the face cordes and means of them 
            for (x1, y1, w, h) in faces:
		'''fcountr+=1
                xycords[fcountr-1,]=[x1,y1]
                if fcountr==10:
                    fcountr=0
                    (x,y)=np.int32(np.mean(xycords,0))
                elif fcountr==1:'''
                (x,y,w,h)=(x1,y1,w,h)
		#cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break
            # find left eye cordes and center 
            try:
                head=gray[y:y+h,x:x+w]
                left_eye_region = head[0.2*h:0.5*h,0.1*w:0.5*w]
                left_eye=left_eye_casc.detectMultiScale(left_eye_region)#scaleFactor=1.1, minNeighbors=3) #flags=cv2.CV_HAAR_FIND_BIGGEST_OBJECT)
                for (xl, yl, wl, hl) in left_eye:
                    # Get the mean location of left eye
                    '''lecout+=1
                    lecormean[lecout-1,]=[xl1, yl1, wl1, hl1]
                    if lecout==5:
                        lecout=0
                        (xl,yl,wl,hl)=np.int32(np.mean(lecormean,0) )
                    elif lecout==1:'''
                       # (xl,yl,wl,hl)=(xl1,yl1,wl1,hl1)
                    #Calc the eye center
                    (lex,ley)=(x+w/10+xl+hl/2,y+h/5+yl+wl/2)
                    break 	
                #Find the right eye cordes and center
                right_eye_region=head[0.2*h:0.5*h,0.5*w:0.9*w]
                right_eye=right_eye_casc.detectMultiScale(right_eye_region)# scaleFactor=1.1, minNeighbors=3)#flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
                for (xr, yr, wr, hr) in right_eye:
                    '''recout+=1
                    recormean[recout-1,]=[xr1, yr1, wr1, hr1]
                    if recout==5:
                        recout=0
                        (xr,yr,wr,hr)=np.int32(np.mean(lecormean,0) )
                    elif recout==1:'''
                        #(xr,yr,wr,hr)=(xl1,yl1,wr1,hr1)
                    (rex,rey)=(x+w/2+xr+wr/2,y+h/5 + yr + hr/2)
                    break
                
                if mood=='Sad':
                    frame=sad(gray,frame,(x,y,w,h),(rex,rey,wr,hr),(lex,ley))
                    
                elif mood=='Angry':
                    frame=angry(gray,frame,(x,y,w,h))
                    
                elif mood=='Happy':     
                    frame=happy(gray,frame,(x,y,w,h))  
            except:
                pass             
                                                     
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.Close()
                break

        cap.release()
        
text=['Sad','Happy','Angry','Sleepy','Indifferent']
face_cascade =cv2.CascadeClassifier(pathh+'\haarcascade_frontalface_default.xml')
left_eye_casc=cv2.CascadeClassifier(pathh+"\haarcascade_lefteye_2splits.xml")
right_eye_casc=cv2.CascadeClassifier(pathh+"\haarcascade_righteye_2splits.xml")
rlines=genlines()
llines=genlines()
xycords=np.zeros((10,2))
lecormean=np.zeros((10,4))
recormean=np.zeros((10,4))
fcountr,lecout,recout=(0,0,0)
#x,y=(0,0)
#(xl,yl,wl,hl)=(0,0,0,0)
#(xr,yr,wr,hr)=(0,0,0,0)
app = wx.App(False)
win=GetBut(None)
win.Show(True)
app.MainLoop() 