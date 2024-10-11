"""
オートエンコーダの出力で，SAMのセグメンテーションの結果を出力させる
入力が画像そのまま，出力がSAMのセグメンテーション画像
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2

# SAMモデルの初期化
DEVICE = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth").to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model=sam)

# オートエンコーダモデルの定義（入力チャンネル数を3に変更）
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
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
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
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

def mask_unite(img, anns):
    if len(anns) == 0:
        return img
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.array([255,255,255])
        img[m] = color_mask
    return img

# SAMを使用したセグメンテーション
# outputのテンソルを受け取り，SAMのセグメンテーションの結果を出力
# テンソルで返す
count = 0
def segment_with_sam(image):
    mask = mask_generator.generate(image)
    mask = mask_unite(image, mask)
    mask = mask.astype(np.uint8)
    return mask

# オートエンコーダの学習
def train_autoencoder(model, train_loader, val_loader, num_epochs, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(image)

            mask_for_sam = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
            mask_for_sam = np.transpose(mask_for_sam, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
            plt.figure(figsize=(12, 6))
            plt.imshow(mask_for_sam[0], cmap='hot')
            plt.axis('off')
            mask_filename = f"output_anomaly/sam3mask_{epoch:02}.jpg"
            plt.savefig(mask_filename, bbox_inches='tight', pad_inches=0)
            
            output_for_sam = (output.detach().cpu().numpy() * 255).astype(np.uint8)
            output_for_sam = np.transpose(output_for_sam, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
            plt.figure(figsize=(12, 6))
            plt.imshow(output_for_sam[0], cmap='hot')
            plt.axis('off')
            output_filename = f"output_anomaly/sam3output_{epoch:02}.jpg"
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)

            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, mask)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

def get_files_in_directory(directory_path):
    # 指定されたディレクトリ内のファイル名を格納するリストを初期化
    files = []
    
    count = 0
    # ディレクトリ内のすべてのファイルとフォルダをリストアップ
    for item in os.listdir(directory_path):
        # アイテムへのフルパスを取得
        item_path = os.path.join(directory_path, item)
        
        # アイテムがファイルであれば、リストに追加
        if os.path.isfile(item_path):
            files.append(item_path)

            count += 1
    
    return files


# メイン処理
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットとデータローダーの準備
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    directory_path = "/mnt/data/datasets/SEM/dataset_v1/normal/"
    sem_datasets = get_files_in_directory(directory_path)
    full_dataset = AnomalyDataset(sem_datasets, transform=transform)

    # データセットを訓練用と検証用に分割（8:2）
    train_rate = 0.8
    train_size = int(train_rate * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # オートエンコーダモデルの初期化と学習
    model = AutoEncoder()
    train_autoencoder(model, train_loader,val_loader, num_epochs=50, device=device)

    # モデルの保存
    torch.save(model.state_dict(), 'autoencoder_model_nosam50.pth')
    print("Model saved successfully.")