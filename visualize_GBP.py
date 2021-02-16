import numpy as np
import os
import pydicom as dcm
import pickle as pkl
import random
import skimage.transform as trans
from shutil import rmtree
import datetime
import sys
import innvestigate as inv 
import innvestigate.utils
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
from model_zoo_keras import *
from utils import *
import argparse as ap

####GLOBALS####
NROWS = 224
NCOLS = 224
NCHANNEL = 1
NCLASS = 3

NSAMPLES = 100

if __name__ == '__main__':
	root = '/scratch/groups/willhies/echo_view_classifier/'
	frame_dir = os.path.join(root, 'frames')

	A4C_dir = os.path.join(frame_dir, 'A4C_frames')
	PLAX_dir = os.path.join(frame_dir, 'PLAX_frames')
	PSAX_dir = os.path.join(frame_dir, 'PSAX_frames')

	weights = '/scratch/groups/willhies/echo_view_classifier/model_weights/Resnet50_021621.hdf5'

	vis_direc = os.path.join(root, 'guided_backprop_vis')
	if os.path.exists(vis_direc):
		rmtree(vis_direc)
	A4C_GBP_direc = os.path.join(vis_direc, 'A4C')
	PLAX_GBP_direc = os.path.join(vis_direc, 'PLAX')
	PSAX_GBP_direc = os.path.join(vis_direc, 'PSAX')
	safe_makedir(A4C_GBP_direc)
	safe_makedir(PLAX_GBP_direc)
	safe_makedir(PSAX_GBP_direc)

	premodel = Resnet50_model(saved_weights=weights, input_size=(NROWS, NCOLS, 1))
	model = inv.utils.model_wo_softmax(premodel) #remove classification layer (needed for GBP)

	model.summary()
	analyzer = inv.analyzer.gradient_based.GuidedBackprop(model) #Create analyzer object with GBP preset for pretty pictures

	A4C_filenames = os.listdir(A4C_dir)
	PLAX_filenames = os.listdir(PLAX_dir)
	PSAX_filenames = os.listdir(PSAX_dir)

	for f in random.sample(A4C_filenames, NSAMPLES):
		arr = imread(os.path.join(A4C_dir, f))
		arr = np.reshape(arr, (1, NROWS, NCOLS, 1))
		a = analyzer.analyze(arr)
		a = np.squeeze(a)
		a = a/np.amax(np.abs(a))
		print('Saving analysis as', os.path.join(A4C_GBP_direc, f))
		#imsave(os.path.join(A4C_GBP_direc, f), a)
		plt.imsave(os.path.join(A4C_GBP_direc, f), a, cmap='seismic', vmin=-1, vmax=1)

	for f in random.sample(PLAX_filenames, NSAMPLES):
		arr = imread(os.path.join(PLAX_dir, f))
		arr = np.reshape(arr, (1, NROWS, NCOLS, 1))
		a = analyzer.analyze(arr)
		a = np.squeeze(a)
		a = a/np.amax(np.abs(a))
		print('Saving analysis as', os.path.join(PLAX_GBP_direc, f))
		#imsave(os.path.join(PLAX_GBP_direc, f), a)
		plt.imsave(os.path.join(PLAX_GBP_direc, f), a, cmap='seismic', vmin=-1, vmax=1)

	for f in random.sample(PSAX_filenames, NSAMPLES):
		arr = imread(os.path.join(PSAX_dir, f))
		arr = np.reshape(arr, (1, NROWS, NCOLS, 1))
		a = analyzer.analyze(arr)
		a = np.squeeze(a)
		a = a/np.amax(np.abs(a))
		print('Saving analysis as', os.path.join(PSAX_GBP_direc, f))
		#imsave(os.path.join(PSAX_GBP_direc, f), a)
		plt.imsave(os.path.join(PSAX_GBP_direc, f), a, cmap='seismic', vmin=-1, vmax=1)

	

	
		
		
	





		

