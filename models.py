# Define the CNN Architecture

# Import the required libraries

import tensorflow as tf
from tensorflow.keras import layers


def model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',input_shape=((150,150,3))))
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(256, kernel_size = (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))    

    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Dense(1, activation = 'sigmoid'))
    
    
    
    print(model.summary())
    
    return model
    

    