import cv2
import numpy as np
import scipy.interpolate as scint

'''
def eyeson(gray ,frame ,xr ,yr ,wr ,hr):
    receye_C =frame[y_r -h_r /2:y_r +h_r /2 ,x_r - w_r /2:x r +w r /2]
    gray 1 =gray.copy()
    cv2.ellipse(gray1 ,(xr ,y r +9) ,( 5 *w r /6 , 2 *h r /8) ,0 ,0 ,360 ,(0 ,0 ,0) ,-1)
    receye G =gray1[y r -h r /2:y r +h r /2 ,x r -w r /2:x r +w r /2]
    ret, mask = cv2.threshold(receyeG, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # receyeC=carton(receyeG,receyeC)
    eye = cv2.bitwise_and(receyeC ,receyeC ,mask = mask_inv)

    position_s =[[- 3 *w r /2 ,- 2 *hr] ,[- 7 *w r /4 ,- 1 *h r /4] ,[-wr ,0]
                 ,[-wr ,- 4 *h r /3]  ]  # product([-3*wl/2,-wl],[-2*hl,-hl,0])
    for (ysh ,xsh) in positions:
        newpositio n =frame[y r -h r / 2 +ysh:y r +h r / 2 +ysh ,x r -w r / 2 +xsh:x r +w r / 2 +xsh]
        newposition = cv2.bitwise_and(newposition ,newposition ,mask = mask)
        dst = cv2.add(newposition ,eye)
        frame[y r -h r / 2 +ysh:y r +h r / 2 +ysh ,x r -w r / 2 +xsh:x r +w r / 2 +xsh ] =dst
    # cv2.ellipse(frame,(x,y+9),(4*wl/6,2*hl/8),0,0,360,(255,255,255),-1)
    return frame
'''
def carton(gray, frame):
    img_Mblur = cv2.medianBlur(gray, 7)
    img_edge = cv2.adaptiveThreshold(img_Mblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    # img_edge[:,:,0],img_edge[:,:,1],img_edge[:,:,2] =(R,G,B)
    temp = cv2.bitwise_and(frame, img_edge)
    frame = temp
    return frame


def create_LUT_8UC1(x, y):
    spl = scint.UnivariateSpline(np.array(x), np.array(y))
    return spl(range(256))


def decolorize(frame, x, y, mood):
    xt = 0
    yt = 0
    rows, cols, cln = frame.shape
    frame1 = frame[y - yt:y + rows, x - xt:x + cols]
    c_r, c_g, c_b = cv2.split(frame1)
    if mood == 'sad':
        colmin = [0, 63, 126, 191, 255]
        colmax = [0, 63, 100, 200, 255]
    else:
        # colmin=[0,150,150,192,250] #HAPPY filter
        # colmax=[0,150,168,192,250]
        colmin = [0, 100, 128, 200, 250]  # Happy filtr
        colmax = [0, 100, 128, 250, 250]

    myLUT = create_LUT_8UC1(colmin, colmax)
    c_r = cv2.LUT(c_r, myLUT).astype(np.uint8)
    c_b = cv2.LUT(c_b, myLUT).astype(np.uint8)
    frame1 = cv2.merge((c_r, c_g, c_b))
    frame[y - yt:y + rows, x - xt:x + cols] = frame1
    return frame


def genlines():
    lines = []
    for lenght in [9, 15, 8, 4]:
        points = []
        x, y = 0, 0
        for _ in range(lenght):
            a = np.random.randint(13)
            b = np.random.randint(20)
            x = x + (-1) ** b * a
            c = np.random.randint(20)
            y = y + c
            points.append((x, y))
        lines.append(points)

    return lines

def sad(gray, frame, x, y, w, h, rex, rey, wr, hr, lex, ley):
    rlines = genlines()
    llines = genlines()
    # frame = eyeson(gray, frame, (rex, rey, wr, hr))
    # clines=copy.copy(lines)
    for points in rlines:
        point = np.int32(points)
        point[:, 0] += rex
        point[:, 1] += rey + 8
        cv2.polylines(frame, [point], False, (0, 0, 0), 3)
    for points in llines:
        point = np.int32(points)
        point[:, 0] += lex
        point[:, 1] += ley + 8
        cv2.polylines(frame, [point], False, (0, 0, 0), 3)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = decolorize(frame, (0, 0), 'sad')
    frame = carton(gray, frame)
    return frame
