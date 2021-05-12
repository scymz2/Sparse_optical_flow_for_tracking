#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import math

NUM_OF_FEATURE_POINTS = 3000  # Number of key points
COLOR = (0, 255, 0)  # Green
VIDEO_PATH = 'test_video1.mp4'  # File path of the video
THICKNESS = 1
SIZE = (1600, 900)
ITERATION = 0
POINTS = []


def mouse_click(event, x, y, flags, params):
    global POINTS, img
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append([x, y])

        for point in POINTS:
            cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), 2)
        display_video(img)


def display_video(frame):
    cv2.resizeWindow("frame", SIZE[0], SIZE[1])
    cv2.imshow('frame', frame)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__ == '__main__':

    cap = cv2.VideoCapture(VIDEO_PATH)
    cv2.namedWindow("frame", 0)
    cv2.setMouseCallback("frame", mouse_click)
    box = [0, 0, 0, 0]
    sift = cv2.SIFT_create(nfeatures=NUM_OF_FEATURE_POINTS, sigma=4.0)

    frame = cap.read()

    while True:

        ret, c_frame = cap.read()

        if not ret:
            break

        c_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)

        img = c_frame

        if len(POINTS) > 0:
            for point in POINTS:
                cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), 2)

            old_points = np.float32([point for point in POINTS]).reshape(-1, 1, 2)
            new_points, status, err = cv2.calcOpticalFlowPyrLK(p_gray, c_gray, old_points, None, **lk_params)

            POINTS.clear()
            for point in new_points.reshape(-1, 2):
                POINTS.append([point[0], point[1]])

        p_gray = c_gray

        display_video(img)

        key = cv2.waitKey(20) & 0xFF
        if key == ord(" "):
            cv2.waitKey(0)

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
