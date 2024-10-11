import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2

# SAMモデルの初期化
DEVICE = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth").to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model=sam)

# オートエンコーダモデルの定義
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 6, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# カスタムデータセット
class AnomalyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = segment_with_sam(image)
        
        if self.transform:
            image = self.transform(Image.fromarray(image))
            mask = self.transform(Image.fromarray(mask))
        
        return image, mask

def mask_unite(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    img[:,:,2] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)*255
        img[m] = color_mask
    return img

# SAMを使用したセグメンテーション
def segment_with_sam(image):
    mask = mask_generator.generate(image)
    mask = mask_unite(mask)
    mask = mask.astype(np.uint8)
    return mask

# オートエンコーダの学習
def train_autoencoder(model, train_loader, num_epochs, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            
            input_tensor = torch.cat([image, mask], dim=1)
            
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, input_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

def get_files_in_directory(directory_path):
    # 指定されたディレクトリ内のファイル名を格納するリストを初期化
    files = []
    
    # ディレクトリ内のすべてのファイルとフォルダをリストアップ
    for item in os.listdir(directory_path):
        # アイテムへのフルパスを取得
        item_path = os.path.join(directory_path, item)
        
        # アイテムがファイルであれば、リストに追加
        if os.path.isfile(item_path):
            files.append(item_path)
    
    return files

# メイン処理
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データセットとデータローダーの準備
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    directory_path = "/mnt/data/datasets/SEM/dataset_v1/normal/"
    sem_datasets = get_files_in_directory(directory_path)
    train_dataset = AnomalyDataset(sem_datasets, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # オートエンコーダモデルの初期化と学習
    model = AutoEncoder()
    train_autoencoder(model, train_loader, num_epochs=50, device=device)

    # モデルの保存
    torch.save(model.state_dict(), 'autoencoder_model50.pth')
    print("Model saved successfully.")