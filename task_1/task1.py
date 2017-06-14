#!/usr/bin/python

import os
import sys
import numpy as np
import caffe
import cv2
import scipy.ndimage as nd

DEBUG = False


# acceptable image suffixes
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp', '.ppm', '.pgm')

NET_FILE = os.path.join(os.path.dirname(__file__), "model.prototxt")
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "weights.caffemodel")

TILE_SIZE = 256
PADDING_SIZE = 50

# number of subwindows processed by a network in a batch
# Higher numbers speed up processing (only marginally once BATCH_SIZE > 16)
# The larger the batch size, the more memory is consumed (both CPU and GPU)
BATCH_SIZE=4

LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2

BACKGROUND = 1
COMMENT = 2
DECORATION = 3
COMMENT_DECORATION = 4
TEXT = 5
TEXT_COMMENT = 6
TEXT_DECORATION = 7
TEXT_COMMENT_DECORATION = 8
BOUNDARY = 9

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
		output = np.copy(network.blobs["probs"].data)
		responses.append(output)
		print "Progress %d%%" % int(100 * idx / float(len(ims)))
	return np.concatenate(responses, axis=0)


def predict(network, ims):
	all_outputs = fprop(network, ims)
	predictions = np.argmax(all_outputs, axis=1)
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


def get_out_im(text_mask, comment_mask, decoration_mask):
	out = np.zeros(text_mask.shape, dtype=np.uint8)

	out[text_mask] += 8
	out[decoration_mask] += 4
	out[comment_mask] += 2
	out[out == 0] = 1  # background class cannot overlap

	return out


def get_color_im(text_mask, comment_mask, decoration_mask):
	color_out = np.concatenate([255 * text_mask[:,:,np.newaxis],
								255 * comment_mask[:,:,np.newaxis],
								255 * decoration_mask[:,:,np.newaxis]],
								axis=2)
	return color_out


def conditional_dilation(mask, boundary_mask, dilation_factor):
	'''
	Dilates $mask by a factor of $dilation_factor, but restricts the growth to
		only those pixels in $boundary_mask
	'''
	allowed_mask = np.logical_or(mask, boundary_mask)

	structure = np.ones((dilation_factor, dilation_factor))
	dilated_mask = nd.binary_dilation(mask, structure)

	return np.logical_and(dilated_mask, allowed_mask)


def apply_post_processing(im):
	
	# pull out raw masks
	text_mask = np.logical_or(
		np.logical_or(im == TEXT, im == TEXT_COMMENT),
		np.logical_or(im == TEXT_DECORATION, im == TEXT_COMMENT_DECORATION)
		)
	comment_mask = np.logical_or(
		np.logical_or(im == COMMENT, im == TEXT_COMMENT),
		np.logical_or(im == COMMENT_DECORATION, im == TEXT_COMMENT_DECORATION)
		)
	decoration_mask = np.logical_or(
		np.logical_or(im == DECORATION, im == TEXT_DECORATION),
		np.logical_or(im == COMMENT_DECORATION, im == TEXT_COMMENT_DECORATION)
		)

	if DEBUG:
		raw_out = get_out_im(text_mask, comment_mask, decoration_mask)
		raw_color = get_color_im(text_mask, comment_mask, decoration_mask)
		cv2.imwrite('raw_out.png', raw_out)
		cv2.imwrite('raw_color.png', raw_color)

	# get the (eroded) boundary mask
	boundary_mask = (im == BOUNDARY)
	if DEBUG:
		cv2.imwrite('bound1.png', (255 * boundary_mask).astype(np.uint8))
	structure = np.ones((3,3))
	non_background_mask = (im != BACKGROUND)
	boundary_mask = np.logical_and(
						nd.binary_erosion(non_background_mask, structure),
						boundary_mask)
	if DEBUG:
		cv2.imwrite('bound2.png', (255 * boundary_mask).astype(np.uint8))

	# expand our predictions
	text_mask = conditional_dilation(text_mask, boundary_mask, 5)
	comment_mask = conditional_dilation(comment_mask, boundary_mask, 5)
	decoration_mask = conditional_dilation(decoration_mask, boundary_mask, 5)

	# format the masks into the output format
	out = get_out_im(text_mask, comment_mask, decoration_mask)

	if DEBUG:
		color = get_color_im(text_mask, comment_mask, decoration_mask)
		cv2.imwrite('color.png', color)

	return out


def main(in_image, out_image):
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

	post_processed = apply_post_processing(result)
	
	print "Writing Final Result"
	cv2.imwrite(out_image, post_processed)

	print "Done"
	print "Exiting"


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "USAGE: python task1.py in_image out_image [gpu#]"
		print "\tin_image is the input image to be labeled"
		print "\tout_image is where the output label image will be written to"
		print "\tgpu is an integer device ID to run networks on the specified GPU.  If omitted, CPU mode is used"
		exit(1)
	# only required argument
	in_image = sys.argv[1]

	# attempt to parse an output directory
	out_image = sys.argv[2]

	# use gpu if specified
	try:
		gpu = int(sys.argv[3])
		if gpu >= 0:
			caffe.set_mode_gpu()
			caffe.set_device(gpu)
	except:
		caffe.set_mode_cpu()

	main(in_image, out_image)
	
