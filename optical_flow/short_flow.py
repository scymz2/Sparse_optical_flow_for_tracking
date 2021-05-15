# !/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import math

COLOR = (0, 255, 0)  # Green
THICKNESS = 1
SIZE = (1600, 900)
TRACK_LENGTH = 20
TRACKS = []
fgbg = cv2.createBackgroundSubtractorMOG2()

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
    return math.sqrt((delta_x ** 2) + (delta_y ** 2))


def draw_min_rectangle(image, contours):
    image = np.copy(image)
    for con in contours:
        min_rect = cv2.minAreaRect(con)
        box = cv2.boxPoints(min_rect)
        min_rect = np.int0(box)
        cv2.drawContours(image, [min_rect], 0, COLOR, 1)
    return image

def process_img(img):
    img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Convert to gray image
    # gray = cv2.equalizeHist(gray)                   # Enhance contrast ratio
    black = fgbg.apply(gray)                        # Subtract background
    img = cv2.medianBlur(black, 5)                  # Median filter
    img = cv2.GaussianBlur(img, (5, 5), 75)         # Gaussian filter
    return img


def get_contours(img):
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def run_moving_objects(VIDEO_PATH, NUM_OF_FEATURE_POINTS):
    global TRACKS

    # Read video
    cap = cv2.VideoCapture(VIDEO_PATH)
    _, p_frame = cap.read()

    # Init video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter('trackMoving.avi', fourcc, fps, (width, height))

    # Generate the gray level image for the first frame
    p_gray = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
    p_gray = cv2.GaussianBlur(p_gray, (5, 5), 75)

    #p_gray = cv2.GaussianBlur(p_gray, (5, 5), 75)


    # Initialize sift and background subtractor
    sift = cv2.SIFT_create(nfeatures=NUM_OF_FEATURE_POINTS, sigma=1.3)

    # Find the feature point
    p_kp = sift.detect(p_gray, None)

    # Extract the vector from the KeyPoint
    p_kp = extract_vector(p_kp)

    # Store all the points into track
    if p_kp is not None:
        for (x, y) in np.float32(p_kp).reshape(-1, 2):
            TRACKS.append([(x, y)])

    while True:
        ret, c_frame = cap.read()

        img = c_frame
        mask = np.zeros_like(c_frame)
        # If there is no frame left, quit
        if not ret:
            TRACKS.clear()
            break
        # Generate the gray level image for the current frame
        c_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
        c_gray = cv2.GaussianBlur(c_gray, (5, 5), 75)

        # Calculate the optical flow
        c_kp, status, err = cv2.calcOpticalFlowPyrLK(p_gray, c_gray, p_kp, None, **lk_params)

        new_tracks = []
        for i, (tr, cur, pre, flag) in enumerate(zip(TRACKS, c_kp, p_kp, status)):

            if not flag == 1:
                continue

            x_cur, y_cur = cur.ravel()
            x_pre, y_pre = pre.ravel()
            dist = calculateDist(x_cur, y_cur, x_pre, y_pre)

            # If the displacement less than 0.3, ignore it
            if dist < 0.3:
                continue

            # Store the points that satisfied the requirements
            tr.append((x_cur, y_cur))

            if len(tr) > TRACK_LENGTH:
                del tr[0]

            new_tracks.append(tr)
            cv2.circle(mask, (x_cur, y_cur), 2, COLOR, -1)

        TRACKS = new_tracks

        cv2.polylines(mask, [np.int32(tr) for tr in TRACKS], False, COLOR, 1)

        # capture the moving objects
        new_img = process_img(img)
        cons = get_contours(new_img)
        img = draw_min_rectangle(img, cons)
        img = cv2.add(img, mask)

        # Resize the window and display video
        display_video(img)
        outVideo.write(img)

        # Set the key to stop the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            TRACKS.clear()
            break

        # Update the frames and feature points
        p_gray = c_gray.copy()
        p_kp = np.float32([tr[-1] for tr in TRACKS]).reshape(-1, 1, 2)


    cv2.destroyAllWindows()
    cap.release()
