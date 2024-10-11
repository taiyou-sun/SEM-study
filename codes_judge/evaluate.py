import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import os
from train import CNN # train.pyからモデルの定義をインポート


def evaluate_model(model_path, data_dir, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(),  # グレイスケールに変換
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # グレイスケール画像の平均と標準偏差
    ])

    test_dataset = datasets.ImageFolder(data_dir, data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False) # バッチサイズを調整する必要があるかもしれません

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    data_dir = "data"
    model_path = "model.pth"
    num_classes = 2

    evaluate_model(model_path, data_dir, num_classes)