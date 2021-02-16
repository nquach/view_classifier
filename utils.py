import os
import pydicom as dcm
import numpy as np
import skimage.transform as trans

NROWS = 112  # pixels in row of frame
NCOLS = 112  # pixels in col of frame
NCHANNEL = 1  # number of channels (hold over from when we incorporated optical flow)

'''
Helper function: safe_makedir
Given path, will make all necessary folders to create path to folder

Input:
path = full path to direc you wish to create

Output:
None
'''
def safe_makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

'''
Helper function: resize_img_list
Given list of images, resize all images in list to dim specified by new_size

Input:
img_list = 1D list of numpy arr
new_size = tuple specifying new image size

Output:
list = list of resized images
'''
def resize_img_list(img_list, new_size):
	new_list = []
	for i in range(len(img_list)):
		new_img = trans.resize(img_list[i], new_size)
		new_list.append(new_img)
	return new_list

'''
Function: dicom2imgdict
Converts PyDICOM dataset into a list of frames
Input: 
* imagefile=PyDICOM dataset class
Output: list of pixel arrays in chronological order
'''
def dicom2imglist(imagefile):
	try:
		ds = imagefile
		nrow = int(ds.Rows)
		ncol = int(ds.Columns)
		ArrayDicom = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
		img_list = []
		if len(ds.pixel_array.shape) == 4:  # format is (nframes, nrow, ncol, 3)
			nframes = ds.pixel_array.shape[0]
			R = ds.pixel_array[:, :, :, 0]
			B = ds.pixel_array[:, :, :, 1]
			G = ds.pixel_array[:, :, :, 2]
			gray = (0.2989 * R + 0.5870 * G + 0.1140 * B)
			for i in range(nframes):
				img_list.append(gray[i, :, :])
			return img_list
		elif len(ds.pixel_array.shape) == 3:  # format (nframes, nrow, ncol) (ie in grayscale already)
			nframes = ds.pixel_array.shape[0]
			for i in range(nframes):
				img_list.append(ds.pixel_array[i, :, :])
			return img_list
	except:
		return None



