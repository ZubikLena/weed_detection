#!/usr/bin/python
# Python 2/3 compatibility
# dev by a human

from __future__ import print_function
import cv2
import numpy as np
import time


def filter(mask):
    kernel = np.ones((7, 7), np.uint8)
    erode = cv2.erode(mask, kernel, iterations=1)
    dilate = cv2.dilate(erode, kernel, iterations=2)
    result = cv2.GaussianBlur(dilate, (3, 3), 1)
    return result


def update(*arg):
    start = time.time()
    h0 = cv2.getTrackbarPos('h min', 'control')
    h1 = cv2.getTrackbarPos('h max', 'control')
    s0 = cv2.getTrackbarPos('s min', 'control')
    s1 = cv2.getTrackbarPos('s max', 'control')
    v0 = cv2.getTrackbarPos('v min', 'control')
    v1 = cv2.getTrackbarPos('v max', 'control')
    lower = np.array((h0, s0, v0))
    upper = np.array((h1, s1, v1))
    mask = cv2.inRange(hsv, lower, upper)
    filtered = filter(mask)
    # contours
    contours, hierarchy = cv2.findContours(
        filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    total_contours = len(contours)
    total_area = 0
    contours_error = []
    contours_ok = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("area: ", area)
        total_area += area
        if (area > 4000 and area < 52500):
            contours_ok.append(cnt)
        else:
            print("ERROR AREA: ", area)
            contours_error.append(cnt)

    if len(contours) > 0:
        media = total_area/len(contours)
        print("\tAREA MEDIA: ", media)
    cv2.imshow('original', src)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours_ok]
    with_bb = src.copy()
    for bounding_box in bounding_boxes:
        # x, y, w, h
        x = bounding_box[0]
        y = bounding_box[1]
        w = bounding_box[2]
        h = bounding_box[3]
        color = (255, 0, 0)
        # if horizontal (w>h) increase height, else increase width to fit required aspect ratio
        axis_to_increase = 'x' if w < h else 'y'

        if (axis_to_increase == 'x'):
            w = h
            x = int(x-0.5*(h-w))
        else:
            h = w
            y = int(y-0.5*(w-h))

        start_point = (x, y)
        end_point = (x+w, y+h)
        print(type(start_point))
        print(type(end_point))
        print(start_point)
        print(end_point)
        cv2.rectangle(with_bb, start_point, end_point, color)
    cv2.imshow('bounding boxes', with_bb)
    # time
    stop = time.time()
    diff = stop - start
    t = str("%.3f" % diff)
    fps = str("%.0f" % (1/diff))
    text = "t["+t+"] fps:["+fps+"] AREAS:["+str(len(contours_ok))+"]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    # show
    print("\tTOTAL: ", total_contours)
    print("\tOK: ", len(contours_ok))
    print("\tERRORS: ", len(contours_error))
    cv2.imshow('mask', mask)
    cv2.imshow('filter: ', filtered)


def main():
    cv2.namedWindow('control', 0)
    cv2.createTrackbar('h min', 'control', 30, 255, update)
    cv2.createTrackbar('h max', 'control', 255, 255, update)
    cv2.createTrackbar('s min', 'control', 60, 255, update)
    cv2.createTrackbar('s max', 'control', 255, 255, update)
    cv2.createTrackbar('v min', 'control', 0, 255, update)
    cv2.createTrackbar('v max', 'control', 255, 255, update)
    im = cv2.resize(src, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('control', im)
    update()
    while 1:
        ch = cv2.waitKey(30)
        if (ch == 27):
            break
            cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
        print("parametro:", fn)
    except:
        fn = 'images/db_023.png'

    src = cv2.imread(fn)
    src = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    print("resized shape:", src.shape)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    main()
