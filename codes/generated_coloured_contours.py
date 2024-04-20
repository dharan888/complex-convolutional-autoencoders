import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel

data_path = 'sinps'

def scale(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    return img

datagen = ImageDataGenerator(preprocessing_function=prep_fn)

train_batch = datagen.flow_from_directory(directory=data_path,target_size=(256,256),class_mode='input',shuffle=False,
                                            color_mode= 'grayscale',classes=None,batch_size=32)
images, labels = next(train_batch)

g = Gaussian2DKernel(1)

for i in range(5):
    relevance = scale(images[i,:,:,0])
    heatmap = convolve(relevance, g)
    name = 'smaps/smp_cpa_' + str(i+1) + '.jpeg'
    #plt.figure(figsize=(3,3))
    plt.figure(dpi=600)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(heatmap, cmap='bwr',vmin=0.15,vmax=0.4)
    plt.savefig(name,bbox_inches='tight',pad_inches=0)
    plt.close()



