import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_auc_score
from captum.attr import ShapleyValueSampling

# Загрузка данных MNIST
train_dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root="./data", train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Определение модели
class Net(nn.Module):
    def init(self):
        super(Net, self).init()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 10)

        # Инициализация весов
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# Обучение модели
def train(model, optimizer, criterion, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


# Оценка модели
def evaluate(model, test_loader):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            targets.extend(target.numpy())
            predictions.extend(predicted.numpy())
    auc = roc_auc_score(targets, predictions)
    return auc


# Создание модели
model = Net()

# Проверка наличия параметров в модели

    # Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

    # Обучение модели
for epoch in range(5):
    print(epoch)
    train(model, optimizer, criterion, train_loader)

    # Вычисление вектора Шепли
explainer = ShapleyValueSampling(model)
data, _ = next(iter(test_loader))
attributions = explainer.attribute(data)

    # Оценка модели с использованием AUC
auc = evaluate(model, test_loader)
print("AUC:", auc)

