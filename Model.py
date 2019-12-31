from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
import numpy as np


def FR_Model(img_shape):

    input_img = Input(shape=img_shape) #(32, 32, 3))
    x = Conv2D(64, (3, 3), activation='relu')(input_img)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(output_dim=64, activation='sigmoid')(x)
    x = Dense(output_dim=64, activation='sigmoid')(x)
    x = Dense(output_dim=3, activation='sigmoid')(x)
    FR_model = Model(input_img, x)

    return  FR_model

