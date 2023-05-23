import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

# Загрузка набора данных Iris
iris = load_iris()
X = iris.data
y = iris.target

# Преобразование целевой переменной в бинарный формат
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=0)

# Функция для расчета вектора Шепли
def shapley_value(model, X):
    shapley_vals = np.zeros(X.shape[1])
    baseline = np.zeros_like(X[0])
    m = X.shape[0]
    for i in range(X.shape[1]):
        phi_sum = 0
        for j in range(m):
            perm = np.random.permutation(m)
            X_perm = X.copy()
            X_perm[:, i] = X_perm[perm, i]
            phi = model.predict(X_perm) - model.predict(np.array([baseline]))
            phi_sum += phi[ :, 1] / m
        shapley_vals[i] = phi_sum.mean()
    return shapley_vals

# Вычисление вектора Шепли
shapley = shapley_value(model, X_test)
print("Shapley values:", shapley)

# Оценка точности модели на тестовых данных
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", accuracy)