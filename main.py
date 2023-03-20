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