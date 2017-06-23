import sys
import os
from collections import defaultdict
import cv2
import time
import numpy as np

def pred_to_textline(pred, orig):
    global_threshold = 127
    ret, th = cv2.threshold(pred,global_threshold,255,cv2.THRESH_BINARY)
    connectivity = 4
    ccResults= cv2.connectedComponentsWithStats(th, connectivity, cv2.CV_32S)

    textlines = []
    for label_id in xrange(1, ccResults[0]):

        baseline = ccResults[1]
        tmp = np.full(baseline.shape, 0, dtype=np.uint8)
        tmp[baseline == label_id] = 255
        im2,contours,hierarchy = cv2.findContours(tmp, 1, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue
        cnt = contours[0]
        cnt = np.squeeze(cnt, axis=1)
        if cnt.shape[0] <=4:
            continue
        textlines.append(cnt)

    return textlines
