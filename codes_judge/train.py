import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import os
import math

# モデルの定義 (512x512入力対応)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# データの読み込み
# データの読み込みと分割
def load_data(data_dir, batch_size, validation_split=0.2): # validation_splitを追加
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(),  # グレイスケールに変換
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # グレイスケール画像の平均と標準偏差
    ])

    dataset = datasets.ImageFolder(data_dir, data_transforms) # train/valの区別なく全て読み込み
    num_classes = len(dataset.classes)

    # データセットを訓練用と検証用に分割
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes


if __name__ == "__main__":
    data_dir = "data"
    batch_size = 32
    validation_split = 0.05 # validationの割合
    epochs = 10
    output_model = "model.pth"

    device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = load_data(data_dir, batch_size, validation_split)

    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    min_loss = math.inf

    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                total_loss += criterion(outputs, labels)

            accuracy = 100 * correct / total
            if total_loss < min_loss:
                min_loss = total_loss
                torch.save(model.state_dict(), output_model)
                print(f"Model saved to {output_model}")
            print(f'Epoch [{epoch+1}/{epochs}],Val Loss: {total_loss}, Val Accuracy: {accuracy:.2f}%')
