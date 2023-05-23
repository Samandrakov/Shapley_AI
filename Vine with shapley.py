import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

# Загрузка набора данных Wine
wine = load_wine()
X = wine.data
y = wine.target

# Стандартизация признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
            phi_sum += phi[:, 0] / m
        shapley_vals[i] = phi_sum.mean()
    return shapley_vals

# Вычисление вектора Шепли
shapley = shapley_value(model, X_train)
print("Shapley values:", shapley)

# Оценка точности модели на тестовых данных с учетом вектора Шепли
def evaluate_with_shapley(model, X_test, y_test, shapley):
    # Веса признаков на основе вектора Шепли
    weights = shapley / np.sum(shapley)
    # Умножение каждого признака на его соответствующий вес
    X_test_weighted = X_test * weights
    # Расчет прогнозов модели на взвешенных данных
    y_pred = model.predict_step(X_test_weighted)
    # Расчет точности модели
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    return accuracy

accuracy_with_shapley = evaluate_with_shapley(model, X_test, y_test, shapley)
print("Accuracy with Shapley:", accuracy_with_shapley)