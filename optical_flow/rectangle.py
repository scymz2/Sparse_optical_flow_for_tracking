#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import cv2

NUM_OF_FEATURE_POINTS = 1000  # Number of key points
COLOR = (0, 255, 0)  # Green
VIDEO_PATH = 'test_video2.mp4'  # File path of the video
THICKNESS = 1
SIZE = (1600, 900)
RECTANGLE = []

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def extract_vector(KeyPoints):
    temp_array = np.zeros((NUM_OF_FEATURE_POINTS, 1, 2), dtype='float32')
    for i in range(len(KeyPoints) - 1):
        temp_array[i][0][0] = KeyPoints[i].pt[0]
        temp_array[i][0][1] = KeyPoints[i].pt[1]
    return temp_array


def display_video(frame):
    cv2.namedWindow("frame", 0)
    cv2.resizeWindow("frame", SIZE[0], SIZE[1])
    cv2.imshow('frame', frame)


def select_rectangle(event, x, y, flags, param):
    global ix, iy
    # If left button down, record the position of it
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        RECTANGLE.append((ix, iy))
    # If the left button up, and the selected shape is a rectangle, draw it
    elif event == cv2.EVENT_LBUTTONUP:
        if (ix != x) & (iy != y):
            cv2.rectangle(p_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            RECTANGLE.append((x, y))
        else:
            RECTANGLE.clear()


if __name__ == '__main__':

    # Read video
    cap = cv2.VideoCapture(VIDEO_PATH)
    _, p_frame = cap.read()

    # Generate the gray level image for the first frame
    p_gray = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)

    # Initialize sift
    sift = cv2.SIFT_create(nfeatures=NUM_OF_FEATURE_POINTS, sigma=3.0)

    # # Find the feature point
    # p_kp = sift.detect(p_gray, None)
    #
    # # Extract the vector from the KeyPoint
    # p_kp = extract_vector(p_kp)

    # Create a mask image for drawing optical flow lines
    mask = np.zeros_like(p_frame)

    # Set Mouse event
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', select_rectangle)

    # 首先循环第一帧读取用户所需要确认的区域
    while True:
        display_video(p_frame)
        if cv2.waitKey(20) & 0xFF == ord('s'):
            break
    cv2.destroyAllWindows()

        # Generate the gray level image for the first frame
    temp_p_gray = p_gray[RECTANGLE[0][1]:RECTANGLE[1][1], RECTANGLE[0][0]:RECTANGLE[1][0]]
    p_kp = sift.detect(temp_p_gray, None)
    p_kp = extract_vector(p_kp)


    while True:
        ret, c_frame = cap.read()
        if not ret:
            break
        if len(RECTANGLE) == 2:

            cv2.rectangle(c_frame, (RECTANGLE[0][0], RECTANGLE[0][1]), (RECTANGLE[1][0], RECTANGLE[1][1]), (0, 255, 0), 2)
            # 说明选框不为空，这样我们需要计算选框之内的点
            # 先处理上一帧

            temp_c_frame = c_frame[RECTANGLE[0][1]:RECTANGLE[1][1], RECTANGLE[0][0]:RECTANGLE[1][0]]
            temp_c_gray = cv2.cvtColor(temp_c_frame, cv2.COLOR_BGR2GRAY)

            c_kp, status, err = cv2.calcOpticalFlowPyrLK(temp_p_gray, temp_c_gray, p_kp, None, **lk_params)

            # sift out the feature points that optical flow exists
            c_tracked_kp = c_kp[status == 1]
            p_tracked_kp = p_kp[status == 1]

            # Draw optical flow tracks
            for (cur, pre) in zip(c_tracked_kp, p_tracked_kp):
                x_cur, y_cur = cur.ravel()
                x_pre, y_pre = pre.ravel()
                x_cur = np.float32(x_cur + RECTANGLE[0][0])
                y_cur = np.float32(y_cur + RECTANGLE[0][1])
                x_pre = np.float32(x_pre + RECTANGLE[0][0])
                y_pre = np.float32(y_pre + RECTANGLE[0][1])
                mask = cv2.line(mask, (x_cur, y_cur), (x_pre, y_pre), COLOR, THICKNESS)
            img = cv2.add(c_frame, mask)

            display_video(img)

            # 更新之前帧和当前帧
            temp_p_gray = temp_c_gray.copy()
            p_kp = c_tracked_kp.reshape(-1, 1, 2)


        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

