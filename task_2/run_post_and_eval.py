import streamlined
import sys
import os
import streamlined
from streamlined.post_processing import post_processing
from streamlined.process_pred import process_data

def get_files(folder, extension=None):
    items = {}
    for root, sub_folders, files in os.walk(folder):
        for f in files:
            if not f.endswith(extension):
                continue
            key = f[:-len(extension)].replace("_TEST", "")
            if key in items:
                raise Exception("Error: this assumes no repeating files names: {}".format(f))

            items[key] = os.path.join(root, f)
    return items

if "__main__" == __name__:

    if True:
        original_imgs_dir = sys.argv[1]
        pred_img_dir = sys.argv[2]
        pre_pred_xml_dir = sys.argv[3]
        gt_xml_dir = sys.argv[4]
        txt_path = sys.argv[5]

        original_imgs_dict = get_files(original_imgs_dir, ".jpg")
        pred_img_dict = get_files(pred_img_dir, ".png")
        gt_xml_dict = get_files(gt_xml_dir, ".xml")
        pre_pred_xml_dict = get_files(pre_pred_xml_dir, ".xml")

        params = {
            "original_img_paths": [],
            "pred_img_paths": [],
            "pre_pred_xml_paths": [],
            "gt_xml_paths": [],
            "pred_txt_paths": [],
            "gt_txt_paths": [],
            "post_processing_function": post_processing.pred_to_pts,
            "website_dir": "website",
            "clean_lsts": True,
            "use_java_parser": False,
            "write_region_header": True
        }

        for k in list(original_imgs_dict.keys()):
            params['original_img_paths'].append(original_imgs_dict[k])
            params['pred_img_paths'].append(pred_img_dict[k])
            params['gt_xml_paths'].append(gt_xml_dict[k])
            params['pre_pred_xml_paths'].append(pre_pred_xml_dict[k])
            params['pred_txt_paths'].append(os.path.join(txt_path, k+"_pred.txt"))
            params['gt_txt_paths'].append(os.path.join(txt_path, k+"_gt.txt"))

    else:
        original_imgs_dir = sys.argv[1]
        pred_xml_dir = sys.argv[2]
        gt_xml_dir = sys.argv[3]
        txt_path = sys.argv[4]

        original_imgs_dict = get_files(original_imgs_dir, ".jpg")
        pred_xml_dict = get_files(pred_xml_dir, ".xml")
        gt_xml_dict = get_files(gt_xml_dir, ".xml")

        params = {
            "original_img_paths": [],
            "gt_xml_paths": [],
            "pred_xml_paths": [],
            "pred_txt_paths": [],
            "gt_txt_paths": [],
            "website_dir": "website",
            "clean_lsts": True,
            "use_java_parser": False,
            "write_region_header": True
        }

        for k in list(original_imgs_dict.keys()):
            params['original_img_paths'].append(original_imgs_dict[k])
            params['pred_xml_paths'].append(pred_xml_dict[k])
            params['gt_xml_paths'].append(gt_xml_dict[k])
            params['pred_txt_paths'].append(os.path.join(txt_path, k+"_pred.txt"))
            params['gt_txt_paths'].append(os.path.join(txt_path, k+"_gt.txt"))

    process_data(params)
