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



if __name__ == '__main__':

    cap = cv2.VideoCapture(VIDEO_PATH)
    cv2.namedWindow("frame", 0)
    box = [0, 0, 0, 0]
    sift = cv2.SIFT_create(nfeatures=NUM_OF_FEATURE_POINTS, sigma=4.0)

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

            if cur_kp is None:
                break
            # 筛选出所有的仍能够被追踪的点
            cur_tracked = cur_kp[status == 1]
            pre_tracked = pre_kp[status == 1]

            cur_tracked = cur_kp
            pre_tracked = pre_kp

            # 画出轨迹
            k = 0
            for i, (cur, pre) in enumerate(zip(cur_tracked, pre_tracked)):
                x_cur, y_cur = cur.ravel()
                x_pre, y_pre = pre.ravel()
                dist = calculateDist(x_cur, y_cur, x_pre, y_pre)
                # 如果距离大于1，就不是静止点
                if dist > 0.2:
                    cur_tracked[k] = cur_tracked[i]
                    pre_tracked[k] = pre_tracked[i]
                    k = k + 1

            cur_tracked = cur_tracked[:k]
            pre_tracked = pre_tracked[:k]

            # 绘制跟踪线
            for (cur, pre) in zip(cur_tracked, pre_tracked):
                x_cur, y_cur = cur.ravel()
                x_pre, y_pre = pre.ravel()
                mask = cv2.line(mask, (x_cur, y_cur), (x_pre, y_pre), COLOR, THICKNESS)
                img = cv2.circle(img, (x_cur, y_cur), 3, COLOR, -1)

            img = cv2.add(img, mask)

            # 更新老图
            p_gray = c_gray.copy()
            pre_kp = cur_tracked.reshape(-1, 1, 2)

        # display video
        display_video(img)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            # select region of interest
            box = cv2.selectROI("frame", img, fromCenter=False, showCrosshair=False)

            # 先读取一帧作为老图
            _, p_frame = cap.read()
            p_gray = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)

            #
            temp_frame = p_gray[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])]
            # cv2.imwrite('frame1.jpg', temp_frame)

            # 找到老图的所有特征点
            pre_kp = sift.detect(temp_frame)
            pre_kp = extract_vector(pre_kp)

            pre_kp = np.float32([(kp[0][0] + box[0], kp[0][1] + box[1]) for kp in pre_kp]).reshape(-1, 1, 2)

            # 创建蒙版
            mask = np.zeros_like(img)

        elif key == ord(" "):
            cv2.waitKey(0)

        elif key == ord("q"):
            break


    cv2.destroyAllWindows()
    cap.release()
