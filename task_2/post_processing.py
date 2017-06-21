import sys
import os
from collections import defaultdict
import cv2
import time
import numpy as np


def pred_to_pts(img):
    global_threshold = 127
    slice_size = 25
    small_threshold = 2000

    #img = cv2.cvtColor( color_img, cv2.COLOR_RGB2GRAY )
    ret, th = cv2.threshold(img,global_threshold,255,cv2.THRESH_BINARY)
    connectivity = 4
    s = time.time()
    output= cv2.connectedComponentsWithStats(th, connectivity, cv2.CV_32S)
    baselines = []

    #skip background
    for label_id in xrange(1, output[0]):
        min_x = output[2][label_id][0]
        min_y = output[2][label_id][1]
        max_x = output[2][label_id][2] + min_x
        max_y = output[2][label_id][3] + min_y
        cnt = output[2][label_id][4]

        if cnt < small_threshold:
            continue

        baseline = output[1][min_y:max_y, min_x:max_x]

        pts = []
        x_all, y_all = np.where(baseline == label_id)
        first_idx = y_all.argmin()
        first = (y_all[first_idx]+min_x, x_all[first_idx]+min_y)

        pts.append(first)
        for i in xrange(0, baseline.shape[1], slice_size):
            next_i = i+slice_size
            baseline_slice = baseline[:, i:next_i]

            x, y = np.where(baseline_slice == label_id)
            x_avg = x.mean()
            y_avg = y.mean()
            pts.append((int(y_avg+i+min_x), int(x_avg+min_y)))

        last_idx = y_all.argmax()
        last = (y_all[last_idx]+min_x, x_all[last_idx]+min_y)
        pts.append(last)

        if len(pts) <= 1:
            continue

        baselines.append(pts)
    return baselines
