import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
import joblib

# Генерация синтетического набора данных для классификации
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Разделение набора данных на обучающую, валидационную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Создание и обучение нейронной сети с валидацией
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Предсказание вероятностей принадлежности классам для тестовой выборки
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Вычисление значения ROC-кривой и её площади под кривой (AUC) на валидационной выборке
fpr_val, tpr_val, thresholds_val = roc_curve(y_val, model.predict_proba(X_val)[:, 1])
roc_auc_val = auc(fpr_val, tpr_val)

# Вывод ROC-кривой на валидационной выборке
plt.plot(fpr_val, tpr_val, color='blue', label='Validation ROC curve (AUC = %0.2f)' % roc_auc_val)

# Вычисление значения ROC-кривой и её площади под кривой (AUC) на тестовой выборке
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob)
roc_auc_test = auc(fpr_test, tpr_test)

# Вывод ROC-кривой на тестовой выборке
plt.plot(fpr_test, tpr_test, color='green', label='Test ROC curve (AUC = %0.2f)' % roc_auc_test)

# Вывод случайной ROC-кривой
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
