import sys
import os
from collections import defaultdict
import subprocess
import multiprocessing
from multiprocessing import Pool, Manager
import time

def get_xml_files(folder):
    items = {}
    for root, sub_folders, files in os.walk(folder):
        for f in files:
            if not f.endswith(".xml"):
                continue
            key = f[:-len(".xml")]
            #Hack for hisdb
            if key in items:
                print "Error: this assumes no repeating files names: {}".format(f)
                raw_input()
                continue


            items[key] = os.path.join(root, f)
    return items

def convert_xml(filepath):
    try:
        cmd = "java -jar evaluator/built_jars/convert_xml.jar {}".format(filepath)
        subprocess.check_call(cmd, shell=True)
    except KeyboardInterrupt:
        raise
    except:
        return False

    return True

def attempt_delete(filepath):
    try:
        os.remove(filepath)
    except:
        pass

def parse_java_output(raw_text):
    results = {}
    results['load_success'] = "Everything loaded without errors." in cmd4_out
    prefix_to_find = [
       "Number of groundtruth lines:",
       "Number of hypothesis lines:",
       "Avg (over pages) P value:",
       "Avg (over pages) R value:",
       "Resulting F_1 value:",
       "Number of pages:"
    ]
    for line in cmd4_out.split("\n"):
        line = line.strip()
        for prefix in prefix_to_find:
            if line.startswith(prefix):
                raw_text[prefix] = float(line[len(prefix):])

    return results

if __name__ == "__main__":

    gt_folder = sys.argv[1]
    pred_folder = sys.argv[2]

    gt_items = get_xml_files(gt_folder)
    pred_items = get_xml_files(pred_folder)

    all_paths = []
    all_paths.extend(list(gt_items.values()))
    all_paths.extend(list(pred_items.values()))


    p = Pool(processes=None)
    m = Manager()
    q = m.Queue()
    results = p.map_async(convert_xml, all_paths)

    left = float('inf')
    while not results.ready():
        if left != results._number_left:
            print "Left {}".format(results._number_left)
            left = results._number_left
        time.sleep(1)

    results = list(results.get())

    print "Successfuly Ran {}/{}".format(sum([1 if r==True else 0 for r in results]), len(results))

    pred_file = "pred.lst"
    gt_file = "gt.lst"

    with open(pred_file, 'w') as f:
        for v in pred_items.values():
            f.write(v+"_reg_0.txt\n")

    with open(gt_file, 'w') as f:
        for v in gt_items.values():
            f.write(v+"_reg_0.txt\n")

    cmd = "java -jar evaluator/built_jars/baseline_evaluator.jar {} {} -no_s".format(gt_file, pred_file)
    cmd_out = subprocess.check_output(cmd, shell=True)

    # results = parse_java_output(cmd_out)
    print cmd_out

    #Clean up
    attempt_delete(pred_file)
    attempt_delete(gt_file)

    for v in pred_items.values():
         attempt_delete(v+"_reg_0.txt")

    for v in gt_items.values():
         attempt_delete(v+"_reg_0.txt")
