import os
import numpy as np
import pickle as pkl
import pydicom as dcm
import skimage.transform as trans
import multiprocessing as mp
from utils import *
from imageio import imwrite
from shutil import copyfile
from random import sample

NROWS = 224  # pixels in row of frame
NCOLS = 224  # pixels in col of frame
NCHANNEL = 1  # number of channels (hold over from when we incorporated optical flow)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_label_filetree(filepath):
	safe_makedir(os.path.join(filepath, 'A4C'))
	safe_makedir(os.path.join(filepath, 'PLAX'))
	safe_makedir(os.path.join(filepath, 'PSAX'))

def create_framename(filepath, frame_num):
	dicom_name = ''
	if filepath[-1] == '/':
		dicom_name = os.path.basename(filepath[:-1])
		return (dicom_name[:-4] + '_' + str(frame_num) + '.png')
	else:
		dicom_name = os.path.basename(filepath)
		return (dicom_name[:-4] + '_' + str(frame_num) + '.png')

def dump_frames_helper(source_file, save_direc):
	print(source_file)
	ds = dcm.dcmread(source_file)
	frame_list = dicom2imglist(ds)
	
	if frame_list == None:
		print('ERROR IN DICOM!!!')
		return
	
	if frame_list[0].shape[0:2] != (NROWS, NCOLS):
		print('Found dcm images of size ', frame_list[0].shape, 'RESIZING!')
		new_frame_list = resize_img_list(frame_list, (NROWS, NCOLS))

	stack = np.stack(new_frame_list, axis=0)
	movie = stack/np.amax(stack) 
	#movie = np.reshape(movie, (movie.shape[0], NROWS, NCOLS, 1))
	for i in range(movie.shape[0]):
		frame = movie[i, :, :]
		frame = np.reshape(frame, (NROWS, NCOLS, 1))
		framename = create_framename(source_file, i)
		frame_path = os.path.join(save_direc, framename)
		imwrite(frame_path, frame)
		print('Saved frame of size', frame.shape)

def dump_frames(source_direc, save_direc):
	filenames = os.listdir(source_direc)
	p = mp.Pool()
	for f in filenames:
		if f[-3:] == 'dcm':
			p.apply_async(dump_frames_helper, [os.path.join(source_direc, f), save_direc])
	p.close()
	p.join()

#view_class must be 'A4C', 'PLAX' or 'PSAX'
def split_frames(root, view_class, train_direc, val_direc, test_direc, train_ratio, val_ratio, test_ratio):
	frames_direc = os.path.join(os.path.join(root, 'frames'), view_class + '_frames')

	if (train_ratio + val_ratio + test_ratio) != 1:
		print('ERROR! Train/Val/Test percentages do not add up to 1.0')
		exit(1)
	else:
		filenames = os.listdir(frames_direc)
		remaining_filenames = os.listdir(frames_direc)
		num_files = len(filenames)
		num_train = int(np.floor(num_files * train_ratio))
		num_val = int(np.floor(num_files * val_ratio))
		num_test = int(num_files - num_train - num_val)

		train_filenames = sample(filenames, num_train)
		
		for f in train_filenames:
			remaining_filenames.remove(f)

		val_filenames = sample(remaining_filenames, num_val)

		for f in val_filenames:
			remaining_filenames.remove(f)

		test_filenames = remaining_filenames

		p = mp.Pool()
		for f in train_filenames:
			source = os.path.join(frames_direc, f)
			dest = os.path.join(os.path.join(train_direc, view_class), f)
			print('Copying file', source, 'to', dest)
			p.apply_async(copyfile, [source, dest])

		for f in val_filenames:
			source = os.path.join(frames_direc, f)
			dest = os.path.join(os.path.join(val_direc, view_class), f)
			print('Copying file', source, 'to', dest)
			p.apply_async(copyfile, [source, dest])

		for f in test_filenames:
			source = os.path.join(frames_direc, f)
			dest = os.path.join(os.path.join(test_direc, view_class), f)
			print('Copying file', source, 'to', dest)
			p.apply_async(copyfile, [source, dest])
		p.close()
		p.join()


if __name__ == '__main__':
	root = '/scratch/groups/willhies/echo_view_classifier/'
	A4C_raw = os.path.join(root, 'A4C_dicoms')
	PLAX_raw = os.path.join(root, 'PLAX_dicoms')
	PSAX_raw = os.path.join(root, 'PSAX_dicoms')
	frames_direc = os.path.join(root, 'frames')
	A4C_frames = os.path.join(frames_direc, 'A4C_frames')
	PLAX_frames = os.path.join(frames_direc, 'PLAX_frames')
	PSAX_frames = os.path.join(frames_direc, 'PSAX_frames')
	dataset_direc = os.path.join(root, 'dataset')
	train_direc = os.path.join(dataset_direc, 'train')
	val_direc = os.path.join(dataset_direc, 'val')
	test_direc = os.path.join(dataset_direc, 'test')

	safe_makedir(A4C_frames)
	safe_makedir(PLAX_frames)
	safe_makedir(PSAX_frames)
	safe_makedir(train_direc)
	safe_makedir(val_direc)
	safe_makedir(test_direc)

	
	create_label_filetree(train_direc)
	create_label_filetree(val_direc)
	create_label_filetree(test_direc)

	dump_frames(A4C_raw, A4C_frames)
	dump_frames(PLAX_raw, PLAX_frames)
	dump_frames(PSAX_raw, PSAX_frames)
	
	split_frames(root, 'A4C', train_direc, val_direc, test_direc, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
	split_frames(root, 'PLAX', train_direc, val_direc, test_direc, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
	split_frames(root, 'PSAX', train_direc, val_direc, test_direc, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)



	


