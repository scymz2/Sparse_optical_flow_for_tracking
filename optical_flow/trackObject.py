#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import math

NUM_OF_FEATURE_POINTS = 500  # Number of key points
COLOR = (0, 255, 0)  # Green
THICKNESS = 1
SIZE = (1600, 900)
POINTS = []

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def extract_vector(KeyPoints):
    return np.float32([point.pt for point in KeyPoints]).reshape(-1, 1, 2)


def display_video(frame):
    cv2.resizeWindow("frame", SIZE[0], SIZE[1])
    cv2.imshow('frame', frame)


def calculateDist(x1, y1, x2, y2):
    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)
    return math.sqrt((delta_x ** 2) + (delta_y ** 2))



def run_track_object(VIDEO_PATH):

    cap = cv2.VideoCapture(VIDEO_PATH)
    cv2.namedWindow("frame", 0)
    box = [0, 0, 0, 0]

    # Init video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter('trackObject.avi', fourcc, fps, (width, height))

    sift = cv2.SIFT_create(nfeatures=NUM_OF_FEATURE_POINTS, sigma=1.3)

    while True:
        # Read the first frame and convert it to gray picture
        ret, c_frame = cap.read()

        # If no more, stop
        if not ret:
            break

        c_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
        c_gray = cv2.medianBlur(c_gray, 5)  # Median filter
        c_gray = cv2.GaussianBlur(c_gray, (5, 5), 75)  # Gaussian filter
        img = c_frame

        # remove the points not in box, but there must be a box
        if not (box[2] == 0 & box[3] == 0):
            # find the corresponding points
            cur_kp, status, err = cv2.calcOpticalFlowPyrLK(p_gray, c_gray, pre_kp, None, **lk_params)

            if cur_kp is None:
                box = [0, 0, 0, 0]
                continue
            # sift all the points that could be tracked
            cur_tracked = cur_kp[status == 1]
            pre_tracked = pre_kp[status == 1]

            # draw the tracks
            k = 0
            for i, (cur, pre) in enumerate(zip(cur_tracked, pre_tracked)):
                x_cur, y_cur = cur.ravel()
                x_pre, y_pre = pre.ravel()
                dist = calculateDist(x_cur, y_cur, x_pre, y_pre)
                # if the displacement larger that 0.3, then drop it
                if dist > 0.3:
                    cur_tracked[k] = cur_tracked[i]
                    pre_tracked[k] = pre_tracked[i]
                    k = k + 1

            cur_tracked = cur_tracked[:k]
            pre_tracked = pre_tracked[:k]

            # draw the optical flow lines
            for (cur, pre) in zip(cur_tracked, pre_tracked):
                x_cur, y_cur = cur.ravel()
                x_pre, y_pre = pre.ravel()
                mask = cv2.line(mask, (x_cur, y_cur), (x_pre, y_pre), COLOR, THICKNESS)
                img = cv2.circle(img, (x_cur, y_cur), 3, COLOR, -1)

            img = cv2.add(img, mask)

            # update the old frame
            p_gray = c_gray.copy()
            pre_kp = cur_tracked.reshape(-1, 1, 2)

        # display video
        display_video(img)
        outVideo.write(img)

        key = cv2.waitKey(20) & 0xFF
        if key == ord(" "):
            # select region of interest
            new_box = cv2.selectROI("frame", img, fromCenter=False, showCrosshair=False)

            if not (new_box[2] == 0 & new_box[3] == 0):
                box = new_box
                # get a frame as the old frame
                _, p_frame = cap.read()
                p_gray = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
                p_gray = cv2.medianBlur(p_gray, 5)  # Median filter
                p_gray = cv2.GaussianBlur(p_gray, (5, 5), 75)  # Gaussian filter

                # Get the image of selected region
                temp_frame = p_gray[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])]
                # cv2.imwrite('frame1.jpg', temp_frame)

                # find all the key points in the old frame
                pre_kp = sift.detect(temp_frame)
                pre_kp = extract_vector(pre_kp)

                pre_kp = np.float32([(kp[0][0] + box[0], kp[0][1] + box[1]) for kp in pre_kp]).reshape(-1, 1, 2)

                # create a mask
                mask = np.zeros_like(img)

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
