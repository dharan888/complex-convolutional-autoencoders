import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import os.path


train_path = 'microstructural_images/train'
test_path  = 'microstructural_images/test'

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    return img

datagen = ImageDataGenerator(preprocessing_function=prep_fn)

train_batches = datagen.flow_from_directory(directory=train_path,target_size=(256,256),class_mode='input',
                                            color_mode= 'grayscale',classes=None,batch_size=32)
test_batches  = datagen.flow_from_directory(directory=test_path,target_size=(256,256),class_mode='input',
                                            color_mode= 'grayscale',classes=None,batch_size=32)

input_img = Input(shape=(256, 256, 1))

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(x)

autoencoder = Model(input_img, decoded)
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
autoencoder.compile(optimizer=opt, loss='mean_squared_error')
autoencoder.summary()

autoencoder.fit(train_batches,
                epochs=50,
                validation_data=test_batches)

autoencoder.save('models/autoencoder_microstructure_model.h5')
autoencoder.save_weights('models/autoencoder_microstructure_weights.h5')


