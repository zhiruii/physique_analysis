import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = '../dataset_normalized'
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
EPOCHS      = 100
PATIENCE = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CHECKPOINT = 'models/best_checkpoint.pth'
CLASS_LABELS = ['advanced', 'beginner', 'intermediate']

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean = IMAGENET_MEAN, std=IMAGENET_STD),

])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std= IMAGENET_STD),
])

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        img, label = self.subset[index]
        return self.transform(img), label
    
    def __len__(self):
        return len(self.subset)
    
base_data = datasets.ImageFolder(root = DATASET_DIR)
train_subset, val_subset = random_split(dataset=base_data, lengths=[0.8, 0.2], generator=torch.Generator().manual_seed(123))

train_set = SubsetWithTransform(train_subset, train_transform)
val_set = SubsetWithTransform(val_subset, val_transform)

train_loader = DataLoader(train_set, batch_size= BATCH_SIZE, shuffle= True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


class MyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.Dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.Dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

#Freezing of model.features
for param in model.features.parameters():
    param.requires_grad = False

#Initialling classifier layer
model.classifier = MyClassifier()

#Move to GPU for faster computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.classifier.parameters(), lr = 0.001)

os.makedirs('models', exist_ok=True)
best_val_loss = float('inf')
epochs_no_improvement = 0

#Training loop
for epoch in range(EPOCHS):
    model.train()
    model.features.eval()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        preds = model(x_batch)

        loss = criterion(preds, y_batch)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f'Epoch {epoch + 1:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improvement = 0
        torch.save(model.state_dict(), CHECKPOINT)
    else:
        epochs_no_improvement += 1
        if epochs_no_improvement >= PATIENCE:
            print(f'Early stopping — no val improvement for {PATIENCE} epochs')
            break
model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
print(f'Loaded best checkpoint (val_loss={best_val_loss:.4f})')

correct = 0
total = 0
all_preds = []
all_labels = []
model.eval()

#Validation loop (replace with test_set when there is more data for the split)
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = model(x_batch)
        predictions = preds.argmax(dim = 1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)
        all_preds.append(predictions)
        all_labels.append(y_batch)

accuracy = correct / total
print(f'accuracy (validation accuracy only): {accuracy}')

all_preds = torch.cat(all_preds).cpu().numpy()
all_labels = torch.cat(all_labels).cpu().numpy()

print(classification_report(all_labels, all_preds, target_names=CLASS_LABELS))
cm = confusion_matrix(all_labels, all_preds)
print(cm)


if input('Continue? (Y/N)').upper() == 'Y':
    torch.save(model.state_dict(), 'models/physique_classifier_3class.pth')
    print('Weights saved to models/physique_classifier_3class.pth') 

    dummy = torch.zeros(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy,
        '../frontend/public/model/model.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=17,
    )
    print('ONNX model exported to frontend/public/model/model.onnx')

else:
    print("Export cancelled")