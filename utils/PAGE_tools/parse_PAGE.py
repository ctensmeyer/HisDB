import sys
import os
from os import path, listdir
from os.path import join, isfile
from collections import defaultdict
import xml.etree.ElementTree
import re
import json

REGION_TYPES = ['TextRegion', 'GraphicRegion', 'TableRegion', 'SeparatorRegion', 'ChartRegion', 'ImageRegion']

def extract_points(data_string):
    return [tuple(int(x) for x in v.split(',')) for v in data_string.split()]

# http://stackoverflow.com/a/12946675/3479446
def get_namespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''

def readXMLFile(xml_file):
    print xml_file
    root = xml.etree.ElementTree.parse(xml_file).getroot()
    namespace = get_namespace(root)

    pages = []
    for page in root.findall(namespace+'Page'):
        pages.append(process_page(page, namespace))

    return pages

def add_text_equiv(root):
    new_text_equiv = xml.etree.ElementTree.Element("TextEquiv")
    root.append(new_text_equiv)

    new_unicode = xml.etree.ElementTree.Element("Unicode")
    new_text_equiv.append(new_unicode)

def addBaselines(xml_file, out_file, baselines):
    xml.etree.ElementTree.register_namespace('', "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")
    tree = xml.etree.ElementTree.parse(xml_file)
    root = tree.getroot()
    namespace = get_namespace(root)
    page = root.find(namespace+'Page')
    for region in page.findall(namespace+'TextRegion'):
        if region.attrib['id'] == "region_textline":
            text_equiv = region.find(namespace+"TextEquiv")
            if text_equiv is not None:
                region.remove(text_equiv)
            for i, b in enumerate(baselines):
                new_textline = xml.etree.ElementTree.Element("TextLine")
                new_textline.attrib['id'] = "textline_{}".format(i)
                new_textline.attrib['custom'] = "0"
                region.append(new_textline)

                new_coords = xml.etree.ElementTree.Element("Coords")
                new_textline.append(new_coords)

                new_baseline = xml.etree.ElementTree.Element("Baseline")
                new_textline.append(new_baseline)

                str_baseline = " ".join(["{},{}".format(x,y) for x,y in b])
                new_baseline.attrib['points'] = str_baseline


                # new_coords.attrib['points'] = str_baseline

    tree.write(out_file)

def process_page(page, namespace):

    page_out = {}
    regions = []
    lines = []

    for region in page.findall(namespace+'TextRegion'):

        region_out, region_lines = process_region(region, namespace)

        regions.append(region_out)
        lines += region_lines

    graphic_regions = []
    for region in page.findall(namespace+'GraphicRegion'):
        region_out, region_lines = process_region(region, namespace)
        graphic_regions.append(region_out)

    all_region_types = {}
    for t in REGION_TYPES:
        type_regions = []
        for region in page.findall(namespace+t):
            region_out, region_lines = process_region(region, namespace, find_subregions=True)
            type_regions.append(region_out)

        all_region_types[t] = type_regions

    page_out['regions'] = regions
    page_out['lines'] = lines
    page_out['all_region_types'] = all_region_types
    page_out['graphic_regions'] = graphic_regions

    return page_out

def process_region(region, namespace, find_subregions=False):

    region_out = {}

    coords = region.find(namespace+'Coords')
    region_out['bounding_poly'] = extract_points(coords.attrib['points'])
    region_out['id'] = region.attrib['id']
    region_out['type'] = region.attrib.get('type', '')

    if find_subregions:
        all_region_types = {}
        for t in REGION_TYPES:
            type_regions = []
            for sub_region in region.findall(namespace+t):
                sub_region_out, sub_region_lines = process_region(sub_region, namespace)
                type_regions.append(sub_region_out)
            all_region_types[t] = type_regions
        region_out['subregions'] = all_region_types

    lines = []
    for line in region.findall(namespace+'TextLine'):
        line_out = process_line(line, namespace)
        line_out['region_id'] = region.attrib['id']
        lines.append(line_out)


    return region_out, lines

def process_line(line, namespace):
    errors = []
    line_out = {}

    if 'custom' in line.attrib:
        custom = line.attrib['custom']
        custom = custom.split(" ")
        if "readingOrder" in custom:
            roIdx = custom.index("readingOrder")
            ro = int("".join([v for v in custom[roIdx+1] if v.isdigit()]))
            line_out['read_order'] = ro

    line_out['id'] = line.attrib['id']

    baseline = line.find(namespace+'Baseline')

    if baseline is not None:
        line_out['baseline'] = extract_points(baseline.attrib['points'])
    else:
        errors.append('No baseline')

    coords = line.find(namespace+'Coords')
    line_out['bounding_poly'] = extract_points(coords.attrib['points'])

    ground_truth = line.find(namespace+'TextEquiv').find(namespace+'Unicode').text

    if ground_truth == None or len(ground_truth) == 0:
        errors.append("No ground truth")
        ground_truth = ""

    line_out['ground_truth'] = ground_truth
    if len(errors) > 0:
        line_out['errors'] = errors

    return line_out

    return {"images":images}

if __name__ == '__main__':
    xml_file = sys.argv[1]
    image_file = sys.argv[2]
    xmlFileResult = readXMLFile(xml_file)
    # print type(xmlFileResult)
    # for res in xmlFileResult:
    #     print res.keys()
    #     print res['regions']
    #     raw_input()
    # print xmlFileResult
