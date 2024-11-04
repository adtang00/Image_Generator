import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import Sequential
from keras.initializers import RandomNormal
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, Conv2DTranspose

def load_dataset():
    #ds[0] is pixels, ds[1] are the labels
    ds = tfds.load('oxford_iiit_pet', with_info = True, as_supervised= True, split = 'train')
    ds = ds[0].map(preprocess_images)
    ds = ds.filter(lambda img, lbl: img is not None)
    ds = ds.cache()
    ds = ds.shuffle(60000)
    ds = ds.batch(128)
    ds = ds.prefetch(64)    #reduces bottlenecking 
    #print(ds.as_numpy_iterator().next()[1])

def preprocess_images(img, label):
    try:
        img = tf.image.resize(img, [256, 256])
        img = tf.image.central_crop(img, central_fraction=0.875)
        return tf.cast(img, tf.float32) / 255.0, label
    except tf.errors.InvalidArgumentError:
        return None, None

def build_generator():
    model = Sequential()
    #takes in random values and reshapes to 7x7x128 
    model.add(keras.layers.Dense(7*7*512, input_dim = 128, kernel_initializer=RandomNormal(stddev=0.02) ))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 512)))

    # Conv2DTranspose Block 1
    model.add(keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same'))  # Upsample to 14x14
    model.add(keras.layers.LeakyReLU(0.2))

    # Conv2DTranspose Block 2
    model.add(keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same'))  # Upsample to 28x28
    model.add(keras.layers.LeakyReLU(0.2))
    
    # Conv2DTranspose Block 3
    model.add(keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same'))  # Upsample to 56x56
    model.add(keras.layers.LeakyReLU(0.2))

    # Conv2DTranspose Block 4
    model.add(keras.layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same'))  # Upsample to 112x112
    model.add(keras.layers.LeakyReLU(0.2))

    # Final Conv2DTranspose to reach 224x224
    model.add(keras.layers.Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='sigmoid'))  # 3 channels for RGB
    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(16, 3, input_shape = (224, 224, 3)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, 3))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, 3))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    #model.add(Conv2D(128, 5))
    #model.add(LeakyReLU(0.2))
    #model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model

def training_loop():
    pass

def user_input():
    dataset, info = tfds.load('oxford_iiit_pet', with_info=True, as_supervised=True, split = 'train')
    class_names = info.features['label'].names
    labels = class_names
    labels = [label.lower() for label in labels]
    input_str = input("Enter a word: ")
    input_str = input_str.lower()
    input_str = input_str.split(" ")
    for i in input_str:
        if(i in labels):
            #call generator
            print("valid input")
            pass
    print("Not a valid input")
    return "Not a valid input"



if __name__ == "__main__":
    generator = build_generator()
    print(generator.summary())
    discrminator = build_discriminator()
    print(discrminator.summary())
    pass