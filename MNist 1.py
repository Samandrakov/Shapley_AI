import keras
import tensorflow as tf
import tensorflow.keras

import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
# импорт модели
from tensorflow.keras.models import Sequential
# импорт оптимайзера
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test,y_test) = mnist.load_data()
print(len(x_train),len(y_train),len(x_test),len(y_test))
print(x_train[0].shape,x_train[0].dtype)
print(x_train[0])
print(y_train[0])
im = plt.imshow(x_train[0], cmap='binary')
plt.axis('off')

plt.show(im)

x_train = x_train/x_train.max()
x_test = x_test/x_test.max()

y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dence(32, activation='relu', input_shape=(x_train[0].shape)),
    tf.keras.layers.Dence(64, activation='relu'),
    tf.keras.layers.Dence(128, activation='relu'),
    tf.keras.layers.Dence(256, activation='relu'),
    tf.keras.layers.Dence(512, activation='relu'),
    layers.Flatten(),
    layers.Dence(10, activation='sigmoid'),
])
model.summary()