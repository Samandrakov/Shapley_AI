import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Загрузка сохраненной модели
model = SimpleNet()
model.load_state_dict(torch.load('mnist_model_20.pt'))
model.eval()  # Установка модели в режим инференса

# Загрузка и предобработка тестовых данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# Выбор случайной картинки из тестового датасета
dataiter = iter(test_loader)
images, labels = dataiter.__next__()

# Выполнение предсказания
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
prediction = predicted.item()

# Визуализация картинки и предсказания
plt.imshow(images[0][0], cmap='gray')
plt.title(f'Prediction: {prediction}')
plt.show()