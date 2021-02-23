import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np 
import random
import pickle as pkl
from model_zoo import *
from multiprocessing import cpu_count
from utils import *
from imageio import imread

RECALCULATE = False

NROWS = 224
NCOLS = 224
NCLASS = 3
BATCH_SIZE = 64
EPOCH = 50
DECAY_EPOCHS = 0

MODEL_NAME = 'Resnet50_021621_cleanedset'
CKPT_MONITOR = 'val_loss'
CKPT_MODE = 'min'
LR = 1e-4
MONITOR_PARAM = 'val_loss'
PATIENCE = 2

def create_classmap(dataset_dir):
	A4C_dir = os.path.join(dataset_dir, 'A4C')
	PLAX_dir = os.path.join(dataset_dir, 'PLAX')
	PSAX_dir = os.path.join(dataset_dir, 'PSAX')

	classmap = {}

	A4C_filenames = os.listdir(A4C_dir)
	PLAX_filenames = os.listdir(PLAX_dir)
	PSAX_filenames = os.listdir(PSAX_dir)

	for f in A4C_filenames:
		if f[-3:] == 'png':
			classmap[f] = 0

	for f in PLAX_filenames:
		if f[-3:] == 'png':
			classmap[f] = 2

	for f in PSAX_filenames:
		if f[-3:] == 'png':
			classmap[f] = 1

	return classmap

def create_pathlist(dataset_dir):
	A4C_dir = os.path.join(dataset_dir, 'A4C')
	PLAX_dir = os.path.join(dataset_dir, 'PLAX')
	PSAX_dir = os.path.join(dataset_dir, 'PSAX')

	path_list = []

	A4C_filenames = os.listdir(A4C_dir)
	PLAX_filenames = os.listdir(PLAX_dir)
	PSAX_filenames = os.listdir(PSAX_dir)

	for f in A4C_filenames:
		if f[-3:] == 'png':
			path_list.append(os.path.join(A4C_dir, f))

	for f in PLAX_filenames:
		if f[-3:] == 'png':
			path_list.append(os.path.join(PLAX_dir, f))

	for f in PSAX_filenames:
		if f[-3:] == 'png':
			path_list.append(os.path.join(PSAX_dir, f))

	return path_list


if __name__ == '__main__':
	root = '/scratch/groups/willhies/echo_view_classifier/'

	ckpt_dir = os.path.join(root, 'model_weights')

	ckpt_path = os.path.join(ckpt_dir, MODEL_NAME + '.hdf5')

	init_epoch_path = os.path.join(ckpt_dir, MODEL_NAME + '_epoch.pkl')
	init_epoch = 0

	if not RECALCULATE:
		if os.path.exists(init_epoch_path):
			init_epoch = int(pkl.load(open(init_epoch_path, 'rb')))

	print('Initial epoch:', init_epoch)

	weights = None
	if os.path.exists(ckpt_path):
		print('Loaded checkpoint ' + ckpt_path + '!')
		weights = ckpt_path
	else:
		print('No checkpoint found, training from scratch!')

	test_dir = '/scratch/groups/willhies/echo_view_classifier/dataset/val/'

	classmap = create_classmap(test_dir)

	
	model = Resnet50_model(saved_weights = weights, input_size=(NROWS, NCOLS, 1), lr=LR, decay_epochs=DECAY_EPOCHS)

	test_paths = create_pathlist(test_dir)

	probs_list = []
	preds_list = []
	truth_list = []
	correct_bool_list = []
	file_list = []

	for f in test_paths:
		if f[-3:] == 'png':
			img = imread(f)

			filename = os.path.basename(f)
			
			if np.amax(img) > 1:
				#mg = img/255.0
				img = img/np.amax(img)

			img = np.reshape(img, (1, NROWS, NCOLS, 1))

			probs = model.predict(img)
			preds = np.argmax(probs)
			truth = classmap[filename]
			correct_bool = (int(preds) == int(truth))
			print('Filename:', filename, 'Probs:', probs, 'Pred:', preds, 'Truth:', truth, 'Bool:', correct_bool)
			probs_list.append(probs)
			preds_list.append(preds)
			truth_list.append(truth)
			correct_bool_list.append(correct_bool)
			file_list.append(filename)

	preds_dir = os.path.join(root, 'predictions')
	safe_makedir(preds_dir)

	probs_filepath = os.path.join(preds_dir, 'probs.pkl')
	preds_filepath = os.path.join(preds_dir, 'preds.pkl')
	truth_filepath = os.path.join(preds_dir, 'truth.pkl')
	bool_filepath = os.path.join(preds_dir, 'bool.pkl')
	filename_filepath = os.path.join(preds_dir, 'filenames.pkl')
	classmap_filepath = os.path.join(preds_dir, 'classmap.pkl')

	pkl.dump(probs_list, open(probs_filepath, 'wb'))
	pkl.dump(preds_list, open(preds_filepath, 'wb'))
	pkl.dump(truth_list, open(truth_filepath, 'wb'))
	pkl.dump(correct_bool_list, open(bool_filepath, 'wb'))
	pkl.dump(file_list, open(filename_filepath, 'wb'))







