import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np 
import random
import pickle as pkl
from model_zoo import *
from multiprocessing import cpu_count
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import *

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

if __name__ == '__main__':
	root = '/scratch/groups/willhies/echo_view_classifier/'

	ckpt_dir = os.path.join(root, 'model_weights')
	safe_makedir(ckpt_dir)
	ckpt_path = os.path.join(ckpt_dir, MODEL_NAME + '.hdf5')

	init_epoch_path = os.path.join(ckpt_dir, MODEL_NAME + '_epoch.pkl')
	init_epoch = 0

	if not RECALCULATE:
		if os.path.exists(init_epoch_path):
			init_epoch = int(pkl.load(open(init_epoch_path, 'rb')))

	print('Initial epoch:', init_epoch)

	weights = None
	if not RECALCULATE:
		if os.path.exists(ckpt_path):
			print('Loaded checkpoint ' + ckpt_path + '!')
			weights = ckpt_path
		else:
			print('No checkpoint found, training from scratch!')
	else:
		print('RECALCULATE flag is TRUE, training from scratch!')

	#train_aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1)
	train_aug = ImageDataGenerator()
	val_aug = ImageDataGenerator() 
	test_aug = ImageDataGenerator()

	train_gen = train_aug.flow_from_directory(directory='/scratch/groups/willhies/echo_view_classifier/dataset/train/', target_size=(224,224), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
	val_gen = val_aug.flow_from_directory(directory='/scratch/groups/willhies/echo_view_classifier/dataset/val/', target_size=(224,224), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
	test_gen = test_aug.flow_from_directory(directory='/scratch/groups/willhies/echo_view_classifier/dataset/test/', target_size=(224,224), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

	STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
	STEP_SIZE_VAL = val_gen.n // val_gen.batch_size

	model = Resnet50_model(saved_weights = weights, input_size=(NROWS, NCOLS, 1), lr=LR, decay_epochs=DECAY_EPOCHS)

	model_checkpoint = ModelCheckpoint(ckpt_path, monitor=CKPT_MONITOR, verbose=1, save_best_only=True, mode=CKPT_MODE)
	csv_direc = os.path.join(root, 'metrics_csv')
	safe_makedir(csv_direc)
	csv_name = MODEL_NAME + '.csv'
	csv_path = os.path.join(csv_direc, csv_name)

	epoch_logger = LambdaCallback(on_epoch_end=lambda epoch, logs: pkl.dump(epoch, open(init_epoch_path, 'wb')))
	reduce_lr = ReduceLROnPlateau(monitor=MONITOR_PARAM, patient=PATIENCE)
	csv_logger = CSVLogger(csv_path)

	num_cpu = cpu_count()

	scores = model.predict(test_gen)
	#history = model.fit_generator(generator=train_gen, epochs=EPOCH, steps_per_epoch = STEP_SIZE_TRAIN, validation_data=val_gen, validation_steps=STEP_SIZE_VAL, callbacks=[epoch_logger, reduce_lr, csv_logger])
	#history = model.fit(x=train_gen, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, callbacks=[epoch_logger, reduce_lr, csv_logger], use_multiprocessing=True, workers=num_cpu)





