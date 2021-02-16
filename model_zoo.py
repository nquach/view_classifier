import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from resnet import ResnetBuilder

def Resnet50_model(saved_weights = None, input_size=(224,224,1), lr=1e-4, decay_epochs=0):
	base_model = ResNet50(include_top=False, weights=None, input_shape=(224,224,1), pooling='avg')
	base_output = base_model.output
	predictions = Dense(3, activation='softmax')(base_output)
	model = Model(inputs=base_model.input, outputs= predictions)
	if saved_weights == None:
		opt = Adam(lr=lr)
		if decay_epochs > 0:
			opt = Adam(lr=lr, decay=lr / config.NUM_EPOCHS)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', TopKCategoricalAccuracy()])
		model.summary()
		return model
	else:
		model.load_weights(saved_weights)
		opt = Adam(lr=lr)
		if decay_epochs > 0:
			opt = Adam(lr=lr, decay=lr / config.NUM_EPOCHS)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary()
		return model

def Resnet50_model_v2(saved_weights = None, input_size=(224,224,1), nclass=3, lr=1e-4, decay_epochs=0):
	model = ResnetBuilder.build_resnet_50(input_size, nclass)
	if saved_weights == None:
		opt = Adam(lr=lr)
		if decay_epochs > 0:
			opt = Adam(lr=lr, decay=lr / config.NUM_EPOCHS)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary()
		return model
	else:
		model.load_weights(saved_weights)
		opt = Adam(lr=lr)
		if decay_epochs > 0:
			opt = Adam(lr=lr, decay=lr / config.NUM_EPOCHS)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary()
		return model


