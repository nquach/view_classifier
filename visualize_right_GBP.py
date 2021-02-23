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

	pred_dir = os.path.join(root, 'predictions')
	pkl_path = os.path.join(pred_dir, 'rights_path_list.pkl')
	wrongs_path_list = pkl.load(open(pkl_path, 'rb'))


	weights = '/scratch/groups/willhies/echo_view_classifier/model_weights/Resnet50_021621.hdf5'

	vis_direc = os.path.join(root, 'guided_backprop_vis_RIGHTS')
	if os.path.exists(vis_direc):
		rmtree(vis_direc)
	safe_makedir(vis_direc)
	
	premodel = Resnet50_model(saved_weights=weights, input_size=(NROWS, NCOLS, 1))
	model = inv.utils.model_wo_softmax(premodel) #remove classification layer (needed for GBP)

	model.summary()
	analyzer = inv.analyzer.gradient_based.GuidedBackprop(model) #Create analyzer object with GBP preset for pretty pictures

	for path in random.sample(wrongs_path_list, NSAMPLES):
		print('Analyzing file:', path)
		arr = imread(path)
		arr = np.reshape(arr, (1, NROWS, NCOLS, 1))
		a = analyzer.analyze(arr)
		a = np.squeeze(a)
		a = a/np.amax(np.abs(a))
		print('Saving analysis as', os.path.join(vis_direc, os.path.basename(path)))
		plt.imsave(os.path.join(vis_direc, os.path.basename(path)), a, cmap='seismic', vmin=-1, vmax=1)




	
		
		
	





		

