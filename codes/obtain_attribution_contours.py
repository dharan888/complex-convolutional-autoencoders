import cv2
import glob
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.layers import Lambda
from sklearn.preprocessing import MinMaxScaler

imgs_path = 'contour_inputs'
file_list = glob.glob('contour_inputs/*.jpg')
images_data  = []
for file in file_list:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images_data.append(img)
images_data = np.array(images_data)
images_data = np.expand_dims(images_data , -1)

def RMS(x):
   rms = tf.sqrt(tf.math.reduce_sum(tf.square(x),axis=[1,2,3]))
   return rms

def pix_atr(xinput):
    h = 0.1
    grads = tf.math.imag(xinput)/h
    atr = RMS(grads)
    return atr

def xReLU(inputs):
    real_inputs  = tf.math.real(inputs)
    imag_inputs  = tf.math.imag(inputs)
    imag_inputs  = tf.where(real_inputs>0,imag_inputs,0)
    real_inputs  = tf.where(real_inputs>0,real_inputs,0)
    outputs = tf.complex(real_inputs,imag_inputs)
    return outputs
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'xReLU': Activation(xReLU)})

class xConv2D(Conv2D):
    def call(self, inputs):
        real_inputs = tf.math.real(inputs)
        imag_inputs = tf.math.imag(inputs)
        real_outputs = self.convolution_op(real_inputs,self.kernel)+self.bias
        imag_outputs = self.convolution_op(imag_inputs,self.kernel)
        outputs     = tf.complex(real_outputs,imag_outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

class xMaxPool2D(MaxPooling2D):
    def call(self, inputs):
        real_inputs = tf.math.real(inputs)
        imag_inputs = tf.math.imag(inputs)
        pool_size = self.pool_size
        strides = self.strides
        padding = self.padding.upper()
        real_outputs,argmax=tf.nn.max_pool_with_argmax(real_inputs,pool_size,strides,padding,include_batch_in_index=True)
        imag_outputs=tf.reshape(tf.gather(tf.reshape(imag_inputs,[-1]),argmax),tf.shape(real_outputs))
        outputs     = tf.complex(real_outputs,imag_outputs)
        return outputs

class xUpSampling2D(UpSampling2D):
    def call(self, inputs):
        real_inputs = tf.math.real(inputs)
        imag_inputs = tf.math.imag(inputs)
        size = self.size
        real_outputs=UpSampling2D(size)(real_inputs)
        imag_outputs=UpSampling2D(size)(imag_inputs)
        outputs     = tf.complex(real_outputs,imag_outputs)
        return outputs


input_img = Input(shape=(256, 256, 1), dtype='complex64')

x = xConv2D(64, (3, 3), activation='xReLU', padding='same')(input_img)
x = xMaxPool2D((2, 2), padding='same')(x)
x = xConv2D(32, (3, 3), activation='xReLU', padding='same')(x)
x = xMaxPool2D((2, 2), padding='same')(x)
x = xConv2D(16, (3, 3), activation='xReLU', padding='same')(x)
x = xMaxPool2D((2, 2), padding='same')(x)
x = xConv2D(8, (3, 3), activation='xReLU', padding='same')(x)
x = xMaxPool2D((2, 2), padding='same',name='code')(x)

x = xUpSampling2D((2, 2))(x)
x = xConv2D(8, (3, 3), activation='xReLU', padding='same')(x)
x = xUpSampling2D((2, 2))(x)
x = xConv2D(16, (3, 3), activation='xReLU', padding='same')(x)
x = xUpSampling2D((2, 2))(x)
x = xConv2D(32, (3, 3), activation='xReLU',padding='same')(x)
x = xUpSampling2D((2, 2))(x)
x = xConv2D(64, (3, 3), activation='xReLU', padding='same')(x)
decoded = xConv2D(1, (3, 3), activation='xReLU', padding='same')(x)

autoencoder = Model(input_img, decoded)
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
autoencoder.compile(optimizer=opt, loss='mean_squared_error')
autoencoder.summary()

autoencoder.load_weights('models/autoencoder_microstructure_weights.h5')

layer_name = 'code'
intr_model= Model(inputs=autoencoder.input,outputs=autoencoder.get_layer(layer_name).output)
y = Lambda(pix_atr)(intr_model.output)
xModel = Model(input_img, y)

h = 0.1
N = 256
input = images_data[0,:,:,:]/255
input = np.expand_dims(input,axis=0)
pix_atr = np.empty(0,'float64')
k=0
for i in range(2):
  for j in range(2):
    xinp = input.copy().astype('complex64')
    xinp[0,i,j,0]+=1j*h
    xoutputs = xModel(xinp)
    pix_atr  = np.append(pix_atr,np.array(xoutputs),axis=0)
    k=k+1

save_path ='contour_outputs'
tf.keras.utils.save_img(save_path, np.reshape(pix_atr,(2,2)))
