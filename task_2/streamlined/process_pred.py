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

    #Load pred image if avaiable
    pred_img_path = params.get('pred_img_paths', None)
    pred_txt_path = params.get("pred_txt_paths", None)
    pred_img = None
    if pred_img_path is not None:
        pred_img = cv2.imread(params['pred_img_paths'], 0)

    #Used Predictions if pred xml exist
    pred_xml_path = params.get("pred_xml_paths", None)
    pred_bl = None
    if pred_xml_path is not None and os.path.exists(pred_xml_path):
        pred_bl = utils.xml_to_bl(pred_xml_path)
        if pred_txt_path is not None:
            if params.get("use_java_parser", False):
                utils.bl_to_xml(pred_bl, pred_xml_path)
                utils.xml_to_txt(pred_xml_path, pred_txt_path)
            else:
                utils.bl_to_txt(pred_bl, pred_txt_path, True)


    elif pred_img is not None:
        #Create Regioned Baseline - load if xml template exists
        pre_pred_bl = {}
        pre_pred_xml_path = params.get("pre_pred_xml_paths", None)
        if pre_pred_xml_path:
            pre_pred_bl = utils.xml_to_bl(pre_pred_xml_path)

        #Run post processing if a function exits
        post_processing_function = params.get("post_processing_function", None)
        if post_processing_function is not None:
            pred_bl = utils.img_to_bl(pred_img, original_img, post_processing_function, pre_pred_bl)

        #Save to xml
        if pred_xml_path is not None and pred_bl is not None:
            utils.bl_to_xml(pred_bl, pred_xml_path)

        #Save pred txt
        if pred_txt_path is not None and pred_bl is not None:
            if params.get("use_java_parser", False):
                if pred_txt_path is None:
                    pred_xml_path = pred_txt_path+".xml"
                utils.bl_to_xml(pred_bl, pred_xml_path)
                utils.xml_to_txt(pred_xml_path, pred_txt_path)
            else:
                utils.bl_to_txt(pred_bl, pred_txt_path, params.get("write_region_header", True))

    #Create visual images
    website_dir = params.get("website_dir", None)
    pred_bl_pixels = None
    if website_dir is not None and pred_bl is not None:
        pred_bl_pixels = utils.bl_to_pixels(pred_bl, original_img.shape)

    #Read GT xml and parse
    #This should be before
    gt_xml_path = params.get("gt_xml_paths", None)
    gt_bl = None
    if gt_xml_path is not None:
        gt_bl = utils.xml_to_bl(gt_xml_path)

    #Save GT txt
    gt_txt_path = params.get("gt_txt_paths", None)
    if gt_txt_path is not None and gt_bl is not None:
        if params.get("use_java_parser", False):
            if gt_xml_path is None:
                gt_xml_path = gt_txt_path+".xml"
            utils.bl_to_xml(gt_bl, gt_xml_path)
            utils.xml_to_txt(gt_xml_path, gt_txt_path)
        else:
            utils.bl_to_txt(gt_bl, gt_txt_path, params.get("write_region_header", True))

    #Save gt visual
    gt_bl_pixels = None
    if website_dir is not None and gt_bl is not None:
        gt_bl_pixels = utils.bl_to_pixels(gt_bl, original_img.shape)

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

    return {
        "vis_result_path": vis_result_path
    }

def process_single_interrupt(params):
    try:
        return process_single(params)
    except KeyboardInterrupt:
        raise
    except:
        return None

    return None


def process_data(params):

    gt_lsts = None
    if params.get("gt_txt_paths", None) is not None:
        gt_lsts = []

    pred_lsts = None
    if params.get("pred_txt_paths", None) is not None:
        pred_lsts = []

    website_img_paths = []

    all_params = []
    original_img_paths = params["original_img_paths"]
    for i in range(len(original_img_paths)):

        single_params = {}
        single_params['id'] = str(i)
        for k, v in params.iteritems():
            if type(v) == list:
                single_params[k] = v[i]
            else:
                single_params[k] = v

        if pred_lsts is not None:
            pred_lsts.append(params["pred_txt_paths"][i])

        if gt_lsts is not None:
            gt_lsts.append(params["gt_txt_paths"][i])

        all_params.append(single_params)
        # result = process_single(single_params)

    print len(all_params)

    p = Pool(processes=None)
    m = Manager()
    q = m.Queue()
    results = p.map_async(process_single_interrupt, all_params)

    left = float('inf')
    while not results.ready():
        if left != results._number_left:
            print "Left {}".format(results._number_left)
            left = results._number_left
        time.sleep(1)

    for result in results.get():
        website_img_paths.append(result.get('vis_result_path', None))

    scores = None
    if gt_lsts is not None and pred_lsts is not None:
        scores = utils.lsts_to_score(gt_lsts, pred_lsts)

        website_dir = params.get('website_dir', None)
        if website_dir is not None:
            gen_webpage.gen_webpage(website_dir, website_img_paths, scores)

        scores.pop("individual_results")
        print scores

    if params.get("clean_lsts", True):
        if gt_lsts is not None:
            for l in gt_lsts:
                utils._attempt_delete(l)

        if pred_lsts is not None:
            for l in pred_lsts:
                utils._attempt_delete(l)

if __name__ == "__main__":


    params = {
        "original_img_paths": ["e-codices_csg-0018_050_max.jpg"],
        "pred_img_paths": ["e-codices_csg-0018_050_max.png"],
        "pre_pred_xml_paths": ["e-codices_csg-0018_050_max_TEST.xml"],
        "gt_xml_paths": ["e-codices_csg-0018_050_max.xml"],
        "pred_txt_paths": ["pred.txt"],
        "gt_txt_paths": ["gt.txt"],
        "website_dir": "website",
        "post_processing_function": post_processing.pred_to_pts
    }

    process_data(params)
