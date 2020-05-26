import cv2


def draw_on_image(img1, img2, x, y, w, h):

    rows, cols, channels = img2.shape
    # img2 = cv2.resize(img2,(w,h),interpolation = cv2.INTER_CUBIC)
    roi = img1[y:y+cols, x-rows:x]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    img1[y:y+cols, x-rows:x] = dst

    return img1
