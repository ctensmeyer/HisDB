import sys
import os
import streamlined
from streamlined.textline_pred import process_data
from streamlined.post_processing import post_textline
from streamlined.post_processing import brian_task3_post

def brian_wrapper(pred, orig):
    return brian_task3_post.getContours(pred, orig)

def get_files(folder, extension=None, use_test=False):
    items = {}
    for root, sub_folders, files in os.walk(folder):
        for f in files:
            if not f.endswith(extension):
                continue
            # f = os.path.join(root, f)
            key = f[:-len(extension)]

            if "_TEST" in key:
                if not use_test:
                    continue
                key = key.replace("_TEST", "")
            elif use_test:
                continue

            if key in items:
                raise Exception("Error: this assumes no repeating files names: {}".format(f))

            items[key] = os.path.join(root, f)
    return items


if __name__ == "__main__":
    original_img_dir = sys.argv[1]
    pred_img_dir = sys.argv[2]
    pre_pred_xml_dir = sys.argv[3]
    gt_xml_dir = sys.argv[4]
    pred_xml_dir = sys.argv[5]

    original_imgs_dict = get_files(original_img_dir, '.jpg')
    pred_img_dict = get_files(pred_img_dir, '.png')
    pre_pred_xml_dict = get_files(pre_pred_xml_dir, ".xml", use_test=True)
    gt_xml_dict = get_files(gt_xml_dir, ".xml")

    params = {
        "original_img_paths": [],
        "pred_img_paths": [],
        "pre_pred_xml_paths": [],
        "gt_xml_paths": [],
        "pred_xml_paths": [],
        "post_processing_function": brian_wrapper,
        "website_dir": "website"
    }


    for k in list(original_imgs_dict.keys()):

        # if "public-test" not in original_imgs_dict[k]:
        #     continue

        params['original_img_paths'].append(original_imgs_dict[k])
        params['pred_img_paths'].append(pred_img_dict[k])
        params['pre_pred_xml_paths'].append(pre_pred_xml_dict[k])
        params['gt_xml_paths'].append(gt_xml_dict[k])

        xml_filename = os.path.basename(gt_xml_dict[k])
        params['pred_xml_paths'].append(os.path.join(pred_xml_dir, xml_filename))

    process_data(params)
