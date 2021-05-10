# !/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import math

NUM_OF_FEATURE_POINTS = 3000  # Number of key points
COLOR = (0, 255, 0)  # Green
VIDEO_PATH = 'test_video1.mp4'  # File path of the video
THICKNESS = 1
SIZE = (1600, 900)
TRACK_LENGTH = 20
TRACKS = []

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def extract_vector(KeyPoints):
    return np.float32([point.pt for point in KeyPoints]).reshape(-1, 1, 2)


def display_video(frame):
    cv2.namedWindow("frame", 0)
    cv2.resizeWindow("frame", SIZE[0], SIZE[1])
    cv2.imshow('frame', frame)


def calculateDist(x1, y1, x2, y2):
    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)
    len = math.sqrt((delta_x ** 2) + (delta_y ** 2))

    return len


if __name__ == '__main__':

    # Read video
    cap = cv2.VideoCapture(VIDEO_PATH)
    _, p_frame = cap.read()

    # Generate the gray level image for the first frame
    p_gray = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)

    # Initialize sift
    sift = cv2.SIFT_create(nfeatures=NUM_OF_FEATURE_POINTS, sigma=3.0)

    # Find the feature point
    p_kp = sift.detect(p_gray, None)

    # Extract the vector from the KeyPoint
    p_kp = extract_vector(p_kp)

    # Store all the points into track
    if p_kp is not None:
        for (x, y) in np.float32(p_kp).reshape(-1, 2):
            TRACKS.append([(x, y)])


    # Create a mask image for drawing optical flow lines
    mask = np.zeros_like(p_frame)

    while True:
        ret, c_frame = cap.read()

        img = c_frame
        # If there is no frame left, quit
        if not ret:
            break
        # Generate the gray level image for the current frame
        c_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow
        c_kp, status, err = cv2.calcOpticalFlowPyrLK(p_gray, c_gray, p_kp, None, **lk_params)

        new_tracks = []
        for i, (tr, cur, pre, flag) in enumerate(zip(TRACKS, c_kp, p_kp, status)):

            if not flag == 1:
                continue

            x_cur, y_cur = cur.ravel()
            x_pre, y_pre = pre.ravel()
            dist = calculateDist(x_cur, y_cur, x_pre, y_pre)

            # 如果位移小于0.3也不符合
            if dist < 0.3:
                continue

            #保存符合的点
            tr.append((x_cur, y_cur))

            if len(tr) > TRACK_LENGTH:
                del tr[0]

            new_tracks.append(tr)
            cv2.circle(img, (x_cur, y_cur), 2, COLOR, -1)

        TRACKS = new_tracks

        cv2.polylines(img, [np.int32(tr) for tr in TRACKS], False, COLOR, 1)


        # Resize the window and display video
        display_video(img)

        # Set the key to stop the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the frames and feature points
        p_gray = c_gray.copy()
        p_kp = np.float32([tr[-1] for tr in TRACKS]).reshape(-1, 1, 2)


    cv2.destroyAllWindows()
    cap.release()
