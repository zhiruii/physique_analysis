import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DATASET_DIR = '../dataset_normalized'
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
EPOCHS      = 30

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root = 'DATASET_DIR', transform=transform)
model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT)
train_set, test_set = random_split(dataset=dataset, lengths=[0.8, 0.2])

train_loader = DataLoader(train_set, batch_size= BATCH_SIZE, shuffle= True)
test_loader = DataLoader(test_set, batch_size= BATCH_SIZE, shuffle = False)

for param in model.features.parameters():
    param.requires_grad = False

class MyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.Dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1280, 3)

    def forward(self, x):
        x = self.Dropout(x)
        x = self.fc(x)
        return x
model.classifier = MyClassifier()

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.classifier.parameters(), lr = 0.001)

for epoch in range(EPOCHS):
    model.classifier.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        preds = model(x_batch)

        loss = criterion(preds, y_batch)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}: Loss was {running_loss / len(train_loader)}')

correct = 0
total = 0

with torch.no_grad():
    model.eval()
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = model(x_batch)
        predictions = preds.argmax(dim = -1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f'accuracy: {accuracy}')