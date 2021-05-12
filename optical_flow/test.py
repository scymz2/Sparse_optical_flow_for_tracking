import numpy as np
import cv2

cap = cv2.VideoCapture('test_video1.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

def draw_min_rect_circle(img, cnts):  # conts = contours
    img = np.copy(img)

    for cnt in cnts:
        # x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue

        min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        min_rect = np.int0(cv2.boxPoints(min_rect))
        cv2.drawContours(img, [min_rect], 0, (0, 255, 0), 1)  # green

        # (x, y), radius = cv2.minEnclosingCircle(cnt)
        # center, radius = (int(x), int(y)), int(radius)  # center and radius of minimum enclosing circle
        # img = cv2.circle(img, center, radius, (0, 0, 255), 2)  # red
    return img

while(1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    gray = fgbg.apply(gray)

    # 高斯
    gray = cv2.medianBlur(gray, 5)

    gray = cv2.GaussianBlur(gray, (5, 5), 75)




    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    img = draw_min_rect_circle(frame, contours)


    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()



#
# img = cv2.imread("tt.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# #均值滤波
# #gray = cv2.blur(gray, (5, 5))
#
# #高斯滤波
# gray = cv2.GaussianBlur(gray, (5, 5), 75)
#
# #中值滤波
# #gray = cv2.medianBlur(gray, 5)
#
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
# img = draw_min_rect_circle(img, contours)
#
#
#
# cv2.imshow("img", img)
# cv2.waitKey(0)