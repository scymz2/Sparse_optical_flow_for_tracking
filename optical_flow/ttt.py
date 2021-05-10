#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2

NUM_OF_FEATURE_POINTS = 4000  # Number of key points
COLOR = (0, 255, 0)  # Green
VIDEO_PATH = 'test_video1.mp4'  # File path of the video
THICKNESS = 1
SIZE = (1600, 900)
ITERATION = 0

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def extract_vector(KeyPoints):
    temp_array = np.zeros((len(KeyPoints), 1, 2), dtype='float32')
    for i in range(len(KeyPoints)):
        temp_array[i][0][0] = KeyPoints[i].pt[0]
        temp_array[i][0][1] = KeyPoints[i].pt[1]
    return temp_array


def display_video(frame):
    cv2.namedWindow("frame", 0)
    cv2.resizeWindow("frame", SIZE[0], SIZE[1])
    cv2.imshow('frame', frame)


def removeKeyPointNotInBox(keypoint):
    temp = np.zeros((1, 1, 2), dtype='float32')
    for i in range(len(keypoint)):
        position_x = keypoint[i][0][0]
        position_y = keypoint[i][0][1]
        if ((position_x > box[0]) & (position_x < (box[0] + box[2])) & (position_y > box[1]) & (
                position_y < (box[1] + box[3]))):
            temp = np.append(temp, [keypoint[i]], axis=0)

    temp = np.delete(temp, 0, axis=0)
    return temp


if __name__ == '__main__':

    cap = cv2.VideoCapture(VIDEO_PATH)
    box = [0, 0, 0, 0]
    sift = cv2.SIFT_create(nfeatures=NUM_OF_FEATURE_POINTS, sigma=3.0)

    # 先读取一帧作为老图
    _, p_frame = cap.read()
    p_gray = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)

    # 找到老图的所有特征点
    pre_kp = sift.detect(p_gray)
    pre_kp = extract_vector(pre_kp)


    while True:
        # 读取新的一帧 并转化为灰度图像
        ret, c_frame = cap.read()
        if not ret:
            break

        c_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)

        # 初始化展示图像
        img = c_frame

        # 把老图中不在区域内的特征点删掉(前提是有box)
        if not (box[2] == 0 & box[3] == 0):
            # 查找对应点
            cur_kp, status, err = cv2.calcOpticalFlowPyrLK(p_gray, c_gray, pre_kp, None, **lk_params)

            # 筛选出所有的仍能够被追踪的点
            cur_tracked = cur_kp[status == 1]
            pre_tracked = pre_kp[status == 1]

            # 画出轨迹
            for (cur, pre) in zip(cur_tracked, pre_tracked):
                x_cur, y_cur = cur.ravel()
                x_pre, y_pre = pre.ravel()
                if ITERATION % 10 == 0:
                    mask = np.zeros_like(c_frame)
                mask = cv2.line(mask, (x_cur, y_cur), (x_pre, y_pre), COLOR, THICKNESS)

            img = cv2.add(img, mask)

            # 更新老图
            p_gray = c_gray.copy()
            pre_kp = cur_tracked.reshape(-1, 1, 2)


        # draw rectangle
        #(x, y, w, h) = [int(v) for v in box]
        #cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)

        # display video
        display_video(img)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            # select region of interest
            box = cv2.selectROI("frame", img, fromCenter=False, showCrosshair=False)

            pre_kp = removeKeyPointNotInBox(pre_kp)

            # 创建蒙版
            mask = np.zeros_like(img)

        elif key == ord("q"):
            break

        # iteration ++
        ITERATION += 1

    cv2.destroyAllWindows()
    cap.release()














