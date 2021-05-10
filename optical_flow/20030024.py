#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import cv2

NUM_OF_FEATURE_POINTS = 3000        # Number of key points
COLOR = (0, 255, 0)  # Green
VIDEO_PATH = 'test_video3.mp4'      # File path of the video
THICKNESS = 1
SIZE = (1600, 900)
ITERATION = 0

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def extract_vector(KeyPoints):
    temp_array = np.zeros((NUM_OF_FEATURE_POINTS, 1, 2), dtype='float32')
    for i in range(len(KeyPoints)-1):
        temp_array[i][0][0] = KeyPoints[i].pt[0]
        temp_array[i][0][1] = KeyPoints[i].pt[1]
    return temp_array


def display_video(frame):
    cv2.namedWindow("frame", 0)
    cv2.resizeWindow("frame", SIZE[0], SIZE[1])
    cv2.imshow('frame', frame)





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

    # Create a mask image for drawing optical flow lines
    mask = np.zeros_like(p_frame)

    while cap.isOpened():
        ret, c_frame = cap.read()
        # If there is no frame left, quit
        if not ret:
            break
        # Generate the gray level image for the current frame
        c_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow
        c_kp, status, err = cv2.calcOpticalFlowPyrLK(p_gray, c_gray, p_kp, None, **lk_params)

        # sift out the feature points that optical flow exists
        c_tracked_kp = c_kp[status == 1]
        p_tracked_kp = p_kp[status == 1]

        # Draw optical flow tracks
        for (cur, pre) in zip(c_tracked_kp, p_tracked_kp):
            x_cur, y_cur = cur.ravel()
            x_pre, y_pre = pre.ravel()
            if ITERATION % 20 == 0:
                mask = np.zeros_like(c_frame)
            mask = cv2.line(mask, (x_cur, y_cur), (x_pre, y_pre), COLOR, THICKNESS)
        img = cv2.add(c_frame, mask)


        # Resize the window and display video
        display_video(img)

        # Set the key to stop the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the frames and feature points
        p_gray = c_gray.copy()
        p_kp = c_tracked_kp.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()
