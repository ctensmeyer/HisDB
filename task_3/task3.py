#!/usr/bin/python

import os
import sys
import numpy as np
import caffe
import cv2
import scipy.ndimage as nd
import streamlined
from streamlined import utils
from streamlined.textline_pred import process_data
from streamlined.post_processing import brian_task3_post
from streamlined.post_processing import post_textline

DEBUG = True


# acceptable image suffixes
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp', '.ppm', '.pgm')

NET_FILE = os.path.join(os.path.dirname(__file__), "model.prototxt")
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "weights.caffemodel")

TILE_SIZE = 384
PADDING_SIZE = 50

# number of subwindows processed by a network in a batch
# Higher numbers speed up processing (only marginally once BATCH_SIZE > 16)
# The larger the batch size, the more memory is consumed (both CPU and GPU)
BATCH_SIZE=3

LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2

def setup_network():
	network = caffe.Net(NET_FILE, WEIGHTS_FILE, caffe.TEST)
	return network


def fprop(network, ims, batchsize=BATCH_SIZE):
	# batch up all transforms at once
	idx = 0
	responses = list()
	while idx < len(ims):
		sub_ims = ims[idx:idx+batchsize]

		network.blobs["data"].reshape(len(sub_ims), ims[0].shape[2], ims[0].shape[1], ims[0].shape[0])

		for x in range(len(sub_ims)):
			transposed = np.transpose(sub_ims[x], [2,0,1])
			transposed = transposed[np.newaxis, :, :, :]
			network.blobs["data"].data[x,:,:,:] = transposed

		idx += batchsize

		# propagate on batch
		network.forward()
		output = np.copy(network.blobs["prob"].data)
		responses.append(output)
		print "Progress %d%%" % int(100 * idx / float(len(ims)))
	return np.concatenate(responses, axis=0)


def predict(network, ims):
	all_outputs = fprop(network, ims)
	predictions = np.squeeze(all_outputs)
	return predictions


def get_subwindows(im):
	height, width, = TILE_SIZE, TILE_SIZE
	y_stride, x_stride, = TILE_SIZE - (2 * PADDING_SIZE), TILE_SIZE - (2 * PADDING_SIZE)
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)
	ims = list()
	bin_ims = list()
	locations = list()
	y = 0
	y_done = False
	while y  <= im.shape[0] and not y_done:
		x = 0
		if y + height > im.shape[0]:
			y = im.shape[0] - height
			y_done = True
		x_done = False
		while x <= im.shape[1] and not x_done:
			if x + width > im.shape[1]:
				x = im.shape[1] - width
				x_done = True
			locations.append( ((y, x, y + height, x + width), 
					(y + PADDING_SIZE, x + PADDING_SIZE, y + y_stride, x + x_stride),
					 TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (im.shape[0] - height) else MIDDLE),
					  LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (im.shape[1] - width) else MIDDLE) 
			) )
			ims.append(im[y:y+height,x:x+width,:])
			x += x_stride
		y += y_stride

	return locations, ims


def stich_together(locations, subwindows, size):
	output = np.zeros(size, dtype=np.uint8)
	for location, subwindow in zip(locations, subwindows):
		outer_bounding_box, inner_bounding_box, y_type, x_type = location
		y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1
		#print outer_bounding_box, inner_bounding_box, y_type, x_type

		if y_type == TOP_EDGE:
			y_cut = 0
			y_paste = 0
			height_paste = TILE_SIZE - PADDING_SIZE
		elif y_type == MIDDLE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif y_type == BOTTOM_EDGE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - PADDING_SIZE

		if x_type == LEFT_EDGE:
			x_cut = 0
			x_paste = 0
			width_paste = TILE_SIZE - PADDING_SIZE
		elif x_type == MIDDLE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif x_type == RIGHT_EDGE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - PADDING_SIZE

		#print (y_paste, x_paste), (height_paste, width_paste), (y_cut, x_cut)

		output[y_paste:y_paste+height_paste, x_paste:x_paste+width_paste] = subwindow[y_cut:y_cut+height_paste, x_cut:x_cut+width_paste]

	return output


def apply_post_processing(im, original_img, xml_file):
	pre_pred_bl = utils.xml_to_bl(xml_file)
	pred_bl = utils.img_to_bl(im, original_img, brian_task3_post.getContours, pre_pred_bl, "textlines")
	#pred_bl = utils.img_to_bl(im, original_img, post_textline.pred_to_textline, pre_pred_bl, "textlines")
	return pred_bl


def write_results(final_result, xml_file):
	utils.bl_to_xml(final_result, xml_file, "textlines")

def main(in_image, in_xml, out_xml):
	print "Loading Image"
	im = cv2.imread(in_image, cv2.IMREAD_COLOR)

	print "Preprocessing"
	data = 0.003921568 * (im - 127.)

	print "Loading network"
	network = setup_network()

	print "Tiling input"
	locations, subwindows = get_subwindows(data)
	print "Number of tiles: %d" % len(subwindows)

	print "Starting Predictions"
	raw_subwindows = predict(network, subwindows)

	print "Reconstructing whole image from tiles"
	result = stich_together(locations, raw_subwindows, tuple(im.shape[0:2]))
	result = (255 * result).astype(np.uint8)

	low_indices = result < 127
	high_indices = result >= 128
	result[low_indices] = 0
	result[high_indices] = 255

	if DEBUG:
		cv2.imwrite('out.png', result)

	print "Applying Post Processing"
	post_processed = apply_post_processing(result, im, in_xml)

	if DEBUG:
		import json
		with open("out.json", "w") as f:
			json.dump(post_processed, f)
	
	print "Writing Final Result"
	write_results(post_processed, out_xml)

	print "Done"
	print "Exiting"


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "USAGE: python task3.py in_image in_xml out_xml [gpu#]"
		print "\tin_image is the input image to be labeled"
		print "\tin_xml is in PAGE format and gives the TextRegion for baseline detection"
		print "\tout_xml is the resulting XML file in PAGE format giving poly-lines for each detected baseline"
		print "\tgpu is an integer device ID to run networks on the specified GPU.  If omitted, CPU mode is used"
		exit(1)
	in_image = sys.argv[1]
	in_xml = sys.argv[2]
	out_xml = sys.argv[3]

	# use gpu if specified
	try:
		gpu = int(sys.argv[4])
		if gpu >= 0:
			caffe.set_mode_gpu()
			caffe.set_device(gpu)
	except:
		caffe.set_mode_cpu()

	main(in_image, in_xml, out_xml)
	
