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
import pandas as pd

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

	test_aug = ImageDataGenerator()

	test_gen = test_aug.flow_from_directory(directory='/scratch/groups/willhies/echo_view_classifier/dataset/test/', target_size=(224,224), color_mode='grayscale', batch_size=1, class_mode='categorical', shuffle=False)

	STEP_SIZE_TEST = test_gen.n // test_gen.batch_size

	model = Resnet50_model(saved_weights = weights, input_size=(NROWS, NCOLS, 1), lr=LR, decay_epochs=DECAY_EPOCHS)

	num_cpu = cpu_count()

	test_gen.reset()
	pred = model.predict_generator(test_gen, steps=STEP_SIZE_TEST, verbose=1)

	predicted_class_indices = np.argmax(pred, axis=1)

	labels = (test_gen.class_indices)
	labels = dict((v,k) for k,v in labels.items())
	predictions = [labels[k] for k in predicted_class_indices]

	filenames = test_gen.filenames
	results = pd.DataFrame({"Filename":filenames, "Predictions":predictions, labels[0] + ' Prob':pred[:,0], labels[1] + ' Prob':pred[:,1], labels[2] + ' Prob':pred[:,2]})
	preds_dir = os.path.join(root, 'predictions')
	safe_makedir(preds_dir)
	results.to_csv(os.path.join(preds_dir, 'results_with_probs.csv'), index=False)
	#history = model.fit_generator(generator=train_gen, epochs=EPOCH, steps_per_epoch = STEP_SIZE_TRAIN, validation_data=val_gen, validation_steps=STEP_SIZE_VAL, callbacks=[epoch_logger, reduce_lr, csv_logger])
	#history = model.fit(x=train_gen, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, callbacks=[epoch_logger, reduce_lr, csv_logger], use_multiprocessing=True, workers=num_cpu)





