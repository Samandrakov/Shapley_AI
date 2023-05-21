import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import shap
from sklearn.preprocessing import LabelEncoder


# Загрузка датасета MNIST и подготовка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

# Определение модели
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Инициализация модели
model = SimpleNet()

# Загрузка предварительно обученных весов (если доступны)
# model.load_state_dict(torch.load('pretrained_model.pth'))

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Цикл обучения модели
num_epochs = 3
for epoch in range(num_epochs):
    print('Обучение эпохи -', epoch+1)
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Выбор примеров для анализа
sample_indices = [0, 1, 2]
X_train = train_dataset.data.float() / 255
X_sample = X_train[sample_indices].view(len(sample_indices), -1)

# Вычисление вектора Шепли
explainer = shap.DeepExplainer(model, train_dataset[:100])
shap_values = explainer.shap_values()
print('Вектор Шепли', shap_values)
le = LabelEncoder()
y = le.fit_transform(test_dataset)
encoding_scheme = dict(zip(y, test_dataset))
print(encoding_scheme)


# Оценка точности модели на тестовом датасете
correct = 0
total = 0

# Отключение вычисления градиентов
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Сохранение обученной модели
torch.save(model.state_dict(), 'mnist_model_20.pt')
print("Обученная модель сохранена")

accuracy = correct / total
print(f'Accuracy: {accuracy}')
print('Количество верных предсказаний - ', correct)
print("Общее число предсказаний - ", total)

# Вывод значений вектора Шепли
print(shap.summary_plot(shap_values, X_sample, feature_names=list(range(784))))