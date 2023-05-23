import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np






# Загрузка данных
train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

# Разделение данных на тренировочную, контрольную и валидационную выборки
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Определение архитектуры нейронной сети
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


# Инициализация модели и оптимизатора
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 2
train_losses = []
val_losses = []
test_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    print('Эпоха',epoch+1)
    train_loss = 0.0
    val_loss = 0.0
    train_total = 0
    val_total = 0
    train_correct = 0
    val_correct = 0

    # Тренировка модели
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accuracy = 100 * train_correct / train_total

    # #Валидация модели
    # model.eval()
    # with torch.no_grad():
    #     for image, labels in val_loader:
    #         outputs = model(images)
    #         # loss = criterion(outputs, labels)
    #         val_loss F+= labels.size(0)
    #         val_correct += (predicted == labels).sum().item()
    #     val_loss /= len(val_loader.dataset)
    #     val_losses.append(val_loss)
    #     val_accuracy = 100 * val_correct / val_total
    #     val_accuracies.append(val_accuracy)

    # Оценка точности на тестовой выборке
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        predicted_probs = []
        true_labels = []

        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_probs.extend(outputs.numpy())
            true_labels.extend(labels.numpy())

        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)

        # ROC-анализ
        # true_labels = true_labels.reshape(-1, 1)
        # predicted_probs = predicted_probs.reshape(-1, 1)
        # true_labels = np.reshape(true_labels, (-1,1))
        # predicted_probs = np.reshape(predicted_probs, (-1,1))

        predicted_probs = torch.cat(predicted_probs, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        predicted_probs = predicted_probs.numpy()
        true_labels = true_labels.numpy()
    for class_label in range(10):
        
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs[:, 1], pos_label=1)
        roc_auc = roc_auc_score(true_labels, predicted_probs[:, 1])
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        # Вывод результатов
    print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    # print(f'Validation Accuracy: {val_accuracy:.2f}%')
    print('-------------------------')