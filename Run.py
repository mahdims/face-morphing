import numpy as np
import cv2
import interface
import wx 
import sys
from sad import sad
from angry import angry
from happy import happy


class GetBut(interface.MyFrame):

    def __init__(self, parent):

        interface.MyFrame.__init__(self, parent)

    def initialize(self):
        self.Show(True)

    def Quit(self, event):
        self.Close()

    def ShowIcon(self, e):

        mood = text[e.GetId()-1]
        # icon = cv2.imread(pathh+'\Icons\icon%d.png' %e.GetId())
        # icon = cv2.resize(icon,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
        # Get the video from the webcam
        cap = cv2.VideoCapture(0)
        # resize the image
        ret = cap.set(3, 966)
        ret = cap.set(4, 668)
        font = cv2.FONT_HERSHEY_SIMPLEX
        while True:
            # read frame by frame
            ret, frame = cap.read()
            if not cap.isOpened():
                # ret, frame = cap.open()
                sys.exit("The WebCam is not working!!")
            # change the image to gray color
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the object (face) in image by help of classifair
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            # get the face coords, width and height
            if len(faces) == 0:
                cv2.imshow('frame', frame)
                continue

            # For now we just work with the first face in the image
            (x, y, w, h) = faces[0]
            # Draw a rectangle around the face
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # find left eye coords and center

            head = gray[y:y+h, x:x+w]
            # restrict the search area for left eye
            left_eye_region = head[int(0.2*h):int(0.5*h), int(0.1*w):int(0.5*w)]
            left_eye = left_eye_casc.detectMultiScale(left_eye_region) #scaleFactor=1.1, minNeighbors=3) #flags=cv2.CV_HAAR_FIND_BIGGEST_OBJECT)
            if len(left_eye) != 0:
                (xl, yl, wl, hl) = left_eye[0]
            else:
                xl, yl, wl, hl = 0, 0, 0, 0
            # Find the center of left eye
            (lex, ley) = (x+w/10+xl+hl/2, y+h/5+yl+wl/2)
            # Find the right eye coords and center
            right_eye_region = head[int(0.2*h):int(0.5*h),int(0.5*w):int(0.9*w)]
            right_eye = right_eye_casc.detectMultiScale(right_eye_region)# scaleFactor=1.1, minNeighbors=3)#flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
            if len(right_eye) != 0:
                (xr, yr, wr, hr) = right_eye[0]
            else:
                xr, yr, wr, hr = 0, 0, 0, 0
                # Find the center of right eye
            (rex, rey) = (x+w/2+xr+wr/2, y+h/5 + yr + hr/2)

            # Which mood we are?
            if mood == 'Sad':
                frame = sad(gray,frame, x,y,w,h, rex,rey,wr,hr,lex,ley)
            elif mood == 'Angry':
                frame = angry(gray,frame, x,y,w,h)
            elif mood == 'Happy':
                frame = happy(gray,frame,x,y,w,h)

            # Show the frame
            cv2.imshow('frame', frame)
            # Continue until the keyword "q" is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.Close()
                break

        cap.release()


text = ['Sad','Happy','Angry','Sleepy','Indifferent']
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
left_eye_casc = cv2.CascadeClassifier('./haarcascade_lefteye_2splits.xml')
right_eye_casc = cv2.CascadeClassifier('./haarcascade_righteye_2splits.xml')
lecormean = np.zeros((10,4))
recormean = np.zeros((10,4))
fcountr,lecout,recout = (0,0,0)

app = wx.App(False)
win = GetBut(None)
win.Show(True)
app.MainLoop()