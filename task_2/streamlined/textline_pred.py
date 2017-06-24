import sys
import os
import cv2
import numpy as np
import utils
from post_processing import post_processing
import gen_webpage
import multiprocessing
from multiprocessing import Pool, Manager
import time
from collections import defaultdict
import math
import traceback

def make_visual_result(original_img, gt_bl_pixels, pred_bl_pixels):

    original_img = original_img.copy()

    not_gt_bl_pixels = np.logical_not(gt_bl_pixels)
    not_pred_bl_pixels = np.logical_not(pred_bl_pixels)

    tp = np.logical_and(gt_bl_pixels, pred_bl_pixels)
    fp = np.logical_and(not_gt_bl_pixels, pred_bl_pixels)
    fn = np.logical_and(gt_bl_pixels, not_pred_bl_pixels)
    # Not using true negative because it is the background
    # tn = np.logical_and(not_gt_bl_pixels, not_pred_bl_pixels)

    original_img[:,:,0][tp] = 255
    original_img[:,:,1][tp] = 0
    original_img[:,:,2][tp] = 0

    original_img[:,:,0][fp] = 0
    original_img[:,:,1][fp] = 255
    original_img[:,:,2][fp] = 0

    original_img[:,:,0][fn] = 0
    original_img[:,:,1][fn] = 0
    original_img[:,:,2][fn] = 255

    return original_img

def process_single(params):

    id_ = params['id']

    original_img = cv2.imread(params["original_img_paths"])
    pred_img = cv2.imread(params['pred_img_paths'], 0)
    website_dir = params.get("website_dir", None)

    pre_pred_bl = {}
    pre_pred_xml_path = params.get("pre_pred_xml_paths", None)
    if pre_pred_xml_path:
        pre_pred_bl = utils.xml_to_bl(pre_pred_xml_path)

    post_processing_function = params.get("post_processing_function", None)
    if post_processing_function is not None:
        pred_bl = utils.img_to_bl(pred_img, original_img, post_processing_function, pre_pred_bl, "textlines")

    pred_xml_path = params.get("pred_xml_paths", None)
    if pred_xml_path is not None and pred_bl is not None:
        utils.bl_to_xml(pred_bl, pred_xml_path, 'textlines')

    #Save gt visual
    pred_bl_pixels = None
    if website_dir is not None and pred_bl is not None:
        pred_bl_pixels = utils.bl_to_textline_pixels(pred_bl, original_img.shape)

    #Parse gt xml
    gt_xml_path = params.get("gt_xml_paths", None)
    gt_bl = None
    if gt_xml_path is not None:
        gt_bl = utils.xml_to_bl(gt_xml_path)

    score = None
    if gt_xml_path is not None and pred_xml_path is not None:
        score = utils.xmls_to_score(params["original_img_paths"], gt_xml_path, pred_xml_path, id_+".csv")

    #Save gt visual
    gt_bl_pixels = None
    if website_dir is not None and gt_bl is not None:
        gt_bl_pixels = utils.bl_to_textline_pixels(gt_bl, original_img.shape)

        #Create overlay visual and save
        vis_result = None
        vis_result_path = None
        if website_dir is not None and (gt_bl_pixels is not None or pred_bl_pixels is not None):

            img1 = pred_bl_pixels
            img2 = gt_bl_pixels
            if img1 is None:
                img1 = img2

            if img2 is None:
                img2 = img1

            vis_result = make_visual_result(original_img, img1, img2)

            vis_result_path = os.path.join("images", id_+".png")

            full_vis_result_path = os.path.join(website_dir, vis_result_path)
            if not os.path.exists(os.path.dirname(full_vis_result_path)):
                os.makedirs(os.path.dirname(full_vis_result_path))
            cv2.imwrite(full_vis_result_path, vis_result)

    print id_
    print score
    return {
        "vis_result_path": vis_result_path,
        "individual_score": score
    }


def process_single_interrupt(params):
    try:
        return process_single(params)
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
        print params
        return None

    return None

def join_scores(scores):
    d = defaultdict(list)
    for s in scores:
        for item in s:
            val = float(item[1])
            if math.isnan(val):
                continue

            d[item[0]].append(val)

    out_d = {}
    for k, v in d.iteritems():
        out_d[k] = np.mean(v), len(v)

    return out_d

def process_data(params):

    website_img_paths = []

    all_params = []
    original_img_paths = params["original_img_paths"]
    for i in range(len(original_img_paths)):

        single_params = {}
        single_params['id'] = os.path.basename(original_img_paths[i]) + str(i)
        for k, v in params.iteritems():
            if type(v) == list:
                single_params[k] = v[i]
            else:
                single_params[k] = v

        all_params.append(single_params)

    # all_params = all_params[:2]

    p = Pool(processes=None)
    m = Manager()
    q = m.Queue()
    results = p.map_async(process_single_interrupt, all_params)
    print len(all_params)
    left = float('inf')
    while not results.ready():
       if left != results._number_left:
           print "Left {}".format(results._number_left)
           left = results._number_left
       time.sleep(1)

    results = results.get()
    # results = []
    # for single_params in all_params[:2]:
    #     print "-"
    #     r = process_single(single_params)
    #     results.append(r)


    results = [r for r in results if r is not None]
    print len(results)

    net_scores = join_scores([s['individual_score'] for s in results])
    print net_scores
