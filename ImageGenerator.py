import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from keras import Sequential
from keras.initializers import RandomNormal
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def load_dataset(batch_size = 32):
    # Load the dataset with images and labels (as_supervised=True gives (image, label) tuples)
    ds = tfds.load('mnist', as_supervised=False, split='train')
    # Map the preprocessing function to the images (this applies only to the images)
    ds = ds.map(preprocess_images)

    # Cache, shuffle, batch, and prefetch the dataset
    ds = ds.cache()
    ds = ds.shuffle(buffer_size = 6000)  # Shuffle the dataset
    ds = ds.batch(batch_size)  # Batch the data
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Prefetch to avoid bottlenecking during training
    return ds

def preprocess_images(img):
    try:
        img = tf.image.resize(img['image'], [28, 28])
        #img = tf.image.central_crop(img, central_fraction=0.875)
        return tf.cast(img, tf.float32) / 255.0
    except tf.errors.InvalidArgumentError:
        return None

def build_generator(latent_dim=100):
    model = Sequential()

    # Start with a (7, 7, 128) feature map after the Dense layer
    model.add(Dense(7*7*128, input_dim=latent_dim, kernel_initializer=RandomNormal(stddev = 0.02)))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))  # Reshape into a 7x7x128 feature map

    # Upsample to 14x14
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))

    # Upsample to 28x28
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))  
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

#input shape matches output of generator
def build_discriminator(input_shape = (28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(64, kernel_size = 3,strides = 2, input_shape = input_shape, padding = 'same'))
    #model.add(keras.layers.BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size = 3,strides = 2, padding = 'same'))
    #model.add(keras.layers.BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

class image_generator(tf.keras.Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Create attributes for gen and disc
        self.generator = generator
        self.discriminator = discriminator

    def compile(self,*args, **kwargs):
        #Compile with base class
        #Adam as optimizer, BinaryCrossEntropy as loss function
        super().compile(*args, **kwargs)
        self.g_opt = Adam(learning_rate = 0.0002, beta_1=0.5)
        self.d_opt = Adam(learning_rate = 0.0002, beta_1=0.5)
        self.g_loss = BinaryCrossentropy()
        self.d_loss = BinaryCrossentropy()

    def train_step(self, image_batch, batch_size = 32):
        real_images = image_batch
        fake_images = self.generator(tf.random.normal((batch_size, 100, 1)), training = False)
        
        #train discriminator
        with tf.GradientTape() as d_tape:
            #pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training = True)
            yhat_fake = self.discriminator(fake_images, training = True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis = 0)
            
            #create lables for real and fakes images
            #add label smoothing
            y_realfake = tf.concat([0.9 * tf.ones_like(yhat_real), 0.1 * tf.zeros_like(yhat_fake)], axis = 0)

            #calculate loss
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

            #apply backpropogation - nnlearn
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        #Train the generator
        with tf.GradientTape() as g_tape:
            #Generate some new images
            gen_images = self.generator(tf.random.normal((32, 100, 1)), training=True)
            #Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training = False)
            #Calculate loss
            total_g_loss = self.g_loss(tf.ones_like(predicted_labels), predicted_labels)
            
        #Apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss":total_d_loss, "g_loss": total_g_loss}


def train_loop():
    ds = load_dataset()
    generator = build_generator()
    #print(generator.summary())
    discriminator = build_discriminator()
    #print(discriminator.summary())
    image_gen = image_generator(generator, discriminator)
    image_gen.compile()
    image_gen.fit(ds, epochs=10)
    generator.save('generator.keras')
    discriminator.save('discriminator.keras')    
                   

    generated_images = image_gen.generator(tf.random.normal((5, 100, 1)))
    for i in range(generated_images.shape[0]):
        plt.subplot(1, 5, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.show()
    

if __name__ == "__main__":
    train_loop()

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