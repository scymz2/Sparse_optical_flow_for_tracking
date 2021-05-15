#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2

NUM_OF_FEATURE_POINTS = 1000  # Number of key points
COLOR = (0, 255, 0)  # Green
THICKNESS = 1
SIZE = (1600, 900)
ITERATION = 0
POINTS = []


# Mouse click listener
def mouse_click(event, x, y, flags, params):
    global POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append([x, y])


# Show image as a video
def display_video(frame):
    cv2.resizeWindow("frame", SIZE[0], SIZE[1])
    cv2.imshow('frame', frame)


# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def run_track_points(VIDEO_PATH):

    # init
    cap = cv2.VideoCapture(VIDEO_PATH)
    cv2.namedWindow("frame", 0)
    cv2.setMouseCallback("frame", mouse_click)

    # Init video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter('trackPoint.avi', fourcc, fps, (width, height))
    _, frame = cap.read()
    mask = np.zeros_like(frame)
    while True:

        ret, c_frame = cap.read()
        # If there is no more frame, stop
        if not ret:
            POINTS.clear()
            break

        c_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
        img = c_frame

        # If there is more than one point, track it
        if len(POINTS) > 0:
            for point in POINTS:
                cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), 2)

            old_points = np.float32([point for point in POINTS]).reshape(-1, 1, 2)
            new_points, status, err = cv2.calcOpticalFlowPyrLK(p_gray, c_gray, old_points, None, **lk_params)

            old_points = old_points[status == 1]
            new_points = new_points[status == 1]

            # draw lines
            for (cur, pre) in zip(new_points, old_points):
                x_cur, y_cur = cur.ravel()
                x_pre, y_pre = pre.ravel()
                mask = cv2.line(mask, (x_cur, y_cur), (x_pre, y_pre), COLOR, THICKNESS)
            img = cv2.add(img, mask)

            # Update the points with new captured ones
            POINTS.clear()
            for point in new_points.reshape(-1, 2):
                POINTS.append([point[0], point[1]])

        p_gray = c_gray
        display_video(img)
        outVideo.write(img)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            POINTS.clear()
            break

    cv2.destroyAllWindows()
    cap.release()
