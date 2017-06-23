import sys
import os
import json
from PAGE_tools import line_extraction, parse_PAGE
from copy import deepcopy
import numpy as np
import cv2
import subprocess
import xml.etree.cElementTree as ET
import csv

def pts_to_xml_str(baseline):
    baseline_txt = []
    for pt in baseline:
        pt_txt = "{},{}".format(*pt)
        baseline_txt.append(pt_txt)
    return " ".join(baseline_txt)

def bl_to_xml(region, xml_path, eval_type="baselines"):
    if "baselines" not in region:
        raise Exception("This takes in only a single region for now")

    baselines = region['baselines']
    textlines = region['textlines']

    if not os.path.exists(os.path.dirname(xml_path)) and len(os.path.dirname(xml_path)) > 0:
        os.makedirs(os.path.dirname(xml_path))

    root = ET.Element("PcGts")
    tree = ET.ElementTree(element=root)

    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Fotini Simistira"
    ET.SubElement(metadata, "Created").text = "2016-06-17T08:11:03.128Z"
    ET.SubElement(metadata, "LastChange").text = "2017-02-14T13:47:41.020Z"

    page = ET.SubElement(root, "Page")
    page.attrib["imageWidth"] = "0"
    page.attrib['imageHeight'] = "0"
    page.attrib['imageFilename'] = ""

    text_region = ET.SubElement(page, "TextRegion")
    text_region.attrib['id'] = "region_textline"

    region_coords = ET.SubElement(text_region, "Coords")
    region_coords.attrib['points'] = pts_to_xml_str(region.get("bounding_poly", []))

    if eval_type == "baselines":
        for i, baseline in enumerate(region['baselines']):
            text_line = ET.SubElement(text_region, "TextLine")
            text_line.attrib['id'] = "textline_"+str(i)

            text_coords = ET.SubElement(text_line, "Coords")
            text_coords.attrib['points'] = pts_to_xml_str([[0,0],[0,0]])

            el_baseline = ET.SubElement(text_line, "Baseline")
            el_baseline.attrib['points'] = pts_to_xml_str(baseline)

    elif eval_type == "textlines":
        for i, textline in enumerate(region['textlines']):
            text_line = ET.SubElement(text_region, "TextLine")
            text_line.attrib['id'] = "textline_"+str(i)

            text_coords = ET.SubElement(text_line, "Coords")
            text_coords.attrib['points'] = pts_to_xml_str(textline)

            el_baseline = ET.SubElement(text_line, "Baseline")
            el_baseline.attrib['points'] = pts_to_xml_str([[0,0],[0,0]])


    xml_str = ET.tostring(root)
    #Hack for namespacing issue
    xml_str = xml_str.replace("<PcGts>", '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">')

    with open(xml_path, 'w') as f:
        f.write(xml_str)

def xml_to_bl(xml_path, region_name="region_textline"):
    xml_data = parse_PAGE.readXMLFile(xml_path)

    if len(xml_data) > 1:
            raise Exception("Not handling this correctly")

    out_regions = {}
    for i, region in enumerate(xml_data[0]['regions']):
        baselines = []
        textlines = []
        region_data = {
            "baselines": baselines,
            'bounding_poly': region['bounding_poly'],
            "textlines": textlines
        }
        out_regions[region.get('id', str(i))] = region_data

        for i, line in enumerate(xml_data[0]['lines']):
            if line['region_id'] != region['id']:
                    continue

            if "baseline" in line:
                baseline = line['baseline']
                baselines.append(baseline)

            if "bounding_poly" in line:
                textlines.append(line['bounding_poly'])

    return out_regions["region_textline"]

def bl_to_json(baselines, json_path):
    raise Exception("Not implemented")

def json_to_bl(json_path):
    raise Exception("Not implemented")

def _baseline_to_str(baseline):
    baseline_txt = []
    for pt in baseline:
        pt_txt = "{},{}".format(*pt)
        baseline_txt.append(pt_txt)
    return ";".join(baseline_txt)

def bl_to_txt(region, txt_path, use_region_header=False):
    if "baselines" not in region:
        raise Exception("This takes in only a single region for now")

    baselines = region['baselines']

    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path))

    with open(txt_path, 'w') as f:
        if use_region_header:
            f.write(_baseline_to_str(region['bounding_poly']) + "\n\n")

        for baseline in baselines:
            f.write(_baseline_to_str(baseline)+"\n")

def txt_to_bl(txt_path):
    raise Exception("Not implemented")

def _draw_line(img, pts, color=(0,255,255), thickness=1, closed=False):
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], closed, color, thickness=thickness)

def bl_to_pixels(baselines, img_shape, thickness=7):
    img = np.zeros(img_shape[:2], dtype=np.uint8)
    for l in baselines['baselines']:
        _draw_line(img, l, color=255, thickness=thickness)
    return img

def extract_region_mask(img, bounding_poly):
    pts = np.array(bounding_poly, np.int32)

    #http://stackoverflow.com/a/15343106/3479446
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_corners = np.array([pts], dtype=np.int32)

    ignore_mask_color = (255,)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color, lineType=cv2.LINE_8)
    return mask

def bl_to_textline_pixels(bl, img_shape, thickness=2):
    img = np.zeros(img_shape[:2], dtype=np.uint8)
    for l in bl['textlines']:
        _draw_line(img, l, color=255, thickness=thickness, closed=True)
    return img

def txts_to_lst(txt_paths):
    raise Exception("Not implemented")

def xml_to_txt(xml_path, txt_path, mode="java"):
    if mode == "python":
        bl_to_txt(xml_to_bl(xml_path), txt_path)
    elif mode == "java":
        cmd = "java -jar evaluator/built_jars/convert_xml.jar {}".format(xml_path)
        subprocess.check_call(cmd, shell=True)
        os.rename(xml_path+"_reg_0.txt", txt_path)
    else:
        raise Exception("Not supported mode")

def _attempt_delete(filepath):
    try:
        os.remove(filepath)
    except:
        pass

def _parse_java_output(raw_text):
    results = {}
    results['load_success'] = "Everything loaded without errors." in raw_text
    prefix_to_find = [
       "Number of groundtruth lines:",
       "Number of hypothesis lines:",
       "Avg (over pages) P value:",
       "Avg (over pages) R value:",
       "Resulting F_1 value:",
       "Number of pages:"
    ]
    start = None
    end = None
    split_lines = raw_text.split("\n")
    for i, line in enumerate(split_lines):
        line = line.strip()
        for prefix in prefix_to_find:
            if line.startswith(prefix):
                results[prefix] = float(line[len(prefix):])

        if line.startswith("Pagewise evaluation"):
            start = i+1

        if line.startswith("#######Final Evaluation#######"):
            end = i

    individual_results = split_lines[start:end]
    individual_results = [[j.strip() for j in i.replace(",", " ").split()] for i  in individual_results if len(i.strip()) > 0]

    results['individual_results'] = individual_results

    return results

def lsts_to_score(gt_lsts, pred_lsts, gt_file="pred.lst", pred_file="gt.lst"):

    with open(gt_file, 'w') as f:
        for l in gt_lsts:
            f.write(os.path.abspath(l)+'\n')

    with open(pred_file, 'w') as f:
        for l in pred_lsts:
            f.write(os.path.abspath(l)+'\n')

    cmd = "java -jar evaluator/built_jars/baseline_evaluator.jar {} {} -no_s".format(gt_file, pred_file)
    cmd_out = subprocess.check_output(cmd, shell=True)

    # print cmd_out
    results = _parse_java_output(cmd_out)

    _attempt_delete(pred_file)
    _attempt_delete(gt_file)

    return results

def xmls_to_score(original_img_path, gt_xml, pred_xml, tmp_csv_file):
    cmd = "java -jar LineSegmentationEvaluator/out/artifacts/hisdoc_layout_comp_jar/hisdoc-layout-comp.jar {} {} {} {} 0.75 false".format(original_img_path, gt_xml, pred_xml, tmp_csv_file)
    cmd_out = subprocess.check_output(cmd, shell=True)
    # print cmd
    # print cmd_out
    with open(tmp_csv_file) as f:
        reader = csv.reader(f)
        rows = [r for r in reader]

    _attempt_delete(tmp_csv_file)

    ret = zip(*tuple(rows))
    ret = [r for r in ret if len(r[0])>0]
    return ret

def img_to_bl(img, original_img, processing_function, region={}, process_type="baselines"):

    if 'bounding_poly' in region:
        ret_region = deepcopy(region)
        rb = region['bounding_poly']
        sub_img = img[rb[0][1]:rb[2][1], rb[0][0]:rb[2][0]]
        sub_original_img = original_img[rb[0][1]:rb[2][1], rb[0][0]:rb[2][0]]

        baselines = processing_function(sub_img, sub_original_img)
        baselines = [[(b[0]+rb[0][0], b[1]+rb[0][1]) for b in baseline] for baseline in baselines]

    else:
        ret_region = {}
        baselines = processing_function(img, original_img)


    # new_baselines = []
    # for baseline in baselines:
    #     tmp_b = [b for (i,b) in enumerate(baseline) if i%5==0]
    #     new_baselines.append(tmp_b)
    #
    # baselines = new_baselines

    ret_region[process_type] = baselines
    return ret_region
