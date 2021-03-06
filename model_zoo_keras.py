import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

def Resnet50_model(saved_weights = None, input_size=(224,224,1), lr=1e-4, decay_epochs=0):
	base_model = ResNet50(include_top=False, weights=None, input_shape=(224,224,1), pooling='avg')
	base_output = base_model.output
	predictions = Dense(3, activation='softmax')(base_output)
	model = Model(inputs=base_model.input, outputs= predictions)
	if saved_weights == None:
		opt = Adam(lr=lr)
		if decay_epochs > 0:
			opt = Adam(lr=lr, decay=lr / config.NUM_EPOCHS)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary()
		return model
	else:
		model.load_weights(saved_weights, by_name=True)
		opt = Adam(lr=lr)
		if decay_epochs > 0:
			opt = Adam(lr=lr, decay=lr / config.NUM_EPOCHS)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary()
		return model

'''
def Resnet152_model(saved_weights = None, input_size=(224,224,1), lr=1e-4, decay_epochs=0):
	base_model = ResNet152(include_top=False, weights=None, input_shape=(224,224,1), pooling='avg')
	base_output = base_model.output
	predictions = Dense(3, activation='softmax')(base_output)
	model = Model(inputs=base_model.input, outputs= predictions)
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
'''