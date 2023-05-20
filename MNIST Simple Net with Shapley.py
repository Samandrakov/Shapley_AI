import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Загрузка датасета MNIST и подготовка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Определение модели
class SimpleNet(nn.Module):
    def init(self):
        super(SimpleNet, self).init()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Инициализация модели
model = SimpleNet()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Цикл обучения модели
num_epochs = 5
for epoch in range(num_epochs):
    print('epoch', epoch+1)
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Оценка точности обычной модели на тестовом датасете
correct = 0
total = 0

# Отключение вычисления градиентов
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy (Simple Model): {accuracy}')

# Загрузка сохраненной модели с вектором Шепли
shepley_model = SimpleNet()
shepley_model.load_state_dict(torch.load('shepley_model.pt'))
shepley_model.eval()  # Установка модели в режим инференса

# Оценка точности модели с вектором Шепли на тестовом датасете
correct_shepley = 0
total_shepley = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = shepley_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_shepley += labels.size(0)
        correct_shepley += (predicted == labels).sum().item()

accuracy_shepley = correct_shepley / total_shepley
print(f'Accuracy (Shepley Model): {accuracy_shepley}')

# Сравнение точности моделей
print("Сравнение точности:")
print(f"Обычная модель: {accuracy}")
print(f"Модель с вектором Шепли: {accuracy_shepley}")