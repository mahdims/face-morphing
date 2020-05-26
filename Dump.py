def halffacedark(gray,frame,(x,y,w,h)):
    ymarg=30
    xmarg=20
    #gray_inv = 255-gray[y-ymarg:y+h+ymarg,x-xmarg:x+w/2]
    #blur = cv2.GaussianBlur(gray_inv, (21,21), 0, 0)
    blur= cv2.medianBlur(gray[y-ymarg:y+h+ymarg,x-xmarg:x+w/2], 7)
    img_edge = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 2)
    img_edge=cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_edge[:,:,2] =255
    #temp=burn(gray[y-ymarg:y+h+ymarg,x-xmarg:x+w/2], img_edge)
    background=gray[y-ymarg:y+h+ymarg,x-xmarg:x+w/2]
    background=cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    temp=cv2.add(background,img_edge)
    #temp=cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB)
    #temp[:,:,2] =100
    frame[y-ymarg:y+h+ymarg,x-xmarg:x+w/2] =temp
   
    return frame
def halffacecarton(gray,frame,(x,y,w,h)):
    ymarg=50
    xmarg=-10
    img_Mblur = cv2.medianBlur(gray[y-ymarg:y+h+ymarg,x-xmarg:x+w/2], 7)
    img_edge = cv2.adaptiveThreshold(img_Mblur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 1)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_edge[:,:,2] =255
    temp1=frame[y-ymarg:y+h+ymarg,x-xmarg:x+w/2]
    temp=cv2.bitwise_and(temp1, img_edge)
    frame[y-ymarg:y+h+ymarg,x-xmarg:x+w/2]=temp
    return frame
def gettheeye(gray,frame,(x,y,w,h)):
    yrpos=-70
    xrpos=0
    receyeC=frame[y-h/2:y+h/2,x-w/2:x+w/2]
    cv2.ellipse(gray,(x,y+9),(4*wl/6,2*hl/8),0,0,360,(0,0,0),-1)
    receyeG=gray[y-h/2:y+h/2,x-w/2:x+w/2,]
    #cv2.imshow('frame1', receyeG)
    ret, mask = cv2.threshold(receyeG, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    eye = cv2.bitwise_and(receyeC,receyeC,mask = mask_inv)
    for shift in [20,50]:
        newposition=frame[yrpos+y-h/2+shift:yrpos+y+h/2+shift,x-w/2+xrpos:x+w/2+xrpos]
        newposition = cv2.bitwise_and(newposition,newposition,mask = mask)
        dst = cv2.add(newposition,eye)
        frame[yrpos+y-h/2+shift:yrpos+y+h/2+shift,x-w/2+xrpos:x+w/2+xrpos]=dst
    #cv2.ellipse(frame,(x,y+9),(4*wl/6,2*hl/8),0,0,360,(255,255,255),-1)
    return frame


    
    