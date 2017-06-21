
import sys
import cv2
import numpy as np

def get_color_im(text_mask, comment_mask, decoration_mask):
	color_out = np.concatenate([255 * text_mask[:,:,np.newaxis],
								255 * comment_mask[:,:,np.newaxis],
								255 * decoration_mask[:,:,np.newaxis]],
								axis=2)
	return color_out

im = cv2.imread(sys.argv[1], -1)

if im.ndim == 3:
	im = im[:,:,0]

text_mask = np.bitwise_and(im, 8)
decoration_mask = np.bitwise_and(im, 4)
comment_mask = np.bitwise_and(im, 2)

out = get_color_im(text_mask, comment_mask, decoration_mask)

cv2.imwrite(sys.argv[2], out)

