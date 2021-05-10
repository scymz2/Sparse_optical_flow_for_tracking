import numpy as np
import cv2

cap = cv2.VideoCapture('video/test_video2.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = (0,255,0)

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)




# Find the feature point by sift
sift = cv2.SIFT_create(nfeatures=2000, sigma=3.0)
p0 = sift.detect(old_gray, None)

# Transfer from KeyPoints format to Array format
temp_array = np.zeros((2000, 1, 2), dtype='float32')
for i in range(len(p0)-1):
    temp_array[i][0][0] = p0[i].pt[0]
    temp_array[i][0][1] = p0[i].pt[1]

p0 = temp_array









# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color, 2)
        #frame = cv2.circle(frame,(a,b),5,color,-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()