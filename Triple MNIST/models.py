"""
Model definitions for Triple MNIST pipeline.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from utils import IMAGE_SIZE, NOISE_DIM

def create_basic_cnn() -> models.Model:
    """
    Basic CNN with flattened 30-dimensional output (3 digits × 10 classes).
    """
    return models.Sequential([
        layers.Input((IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'), layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(30, activation='softmax')
    ])

def create_multi_head_cnn(filters=(64,128,256), l2_reg=0.001, dropout=0.5) -> models.Model:
    """
    CNN with shared backbone and 3 softmax heads (one per digit).
    """
    inp = layers.Input((IMAGE_SIZE, IMAGE_SIZE, 1))
    x = inp
    for f in filters:
        x = layers.Conv2D(f, 3, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)

    outputs = []
    for i in range(3):
        h = layers.Dense(256, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        h = layers.Dropout(dropout)(h)
        outputs.append(layers.Dense(10, activation='softmax', name=f'digit{i+1}')(h))

    return models.Model(inputs=inp, outputs=outputs)

def build_generator() -> models.Sequential:
    """
    DCGAN generator producing IMAGE_SIZE×IMAGE_SIZE grayscale images.
    """
    gen = models.Sequential()
    gen.add(layers.Input((NOISE_DIM,)))
    gen.add(layers.Dense(21*21*256)); gen.add(layers.LeakyReLU(0.2))
    gen.add(layers.Reshape((21,21,256)))
    for f in (128, 64):
        gen.add(layers.Conv2DTranspose(f, 3, strides=2, padding='same'))
        gen.add(layers.LeakyReLU(0.2))
        gen.add(layers.BatchNormalization())
    gen.add(layers.Conv2D(1, 3, activation='tanh', padding='same'))
    return gen
