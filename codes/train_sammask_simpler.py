"""
オートエンコーダの出力で，SAMのセグメンテーションの結果を出力させる
入力が画像そのまま，出力がSAMのセグメンテーション画像
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DataParallel
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2

# SAMモデルの初期化
DEVICE = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth").to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model=sam,points_per_side = 96, pred_iou_thresh=0.92)

class SimplerCNN(nn.Module):
    def __init__(self):
        super(SimplerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # フィルタ数を16に
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # フィルタ数を32に
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # フィルタ数を1に (マスク)
            nn.Sigmoid()  # 出力を0-1の範囲に
        )

    def forward(self, x):
        return self.features(x)

# カスタムデータセット
class AnomalyDataset(Dataset):
    def __init__(self, image_paths, masks, transform=None):
        self.image_paths = image_paths
        self.masks = masks  # マスクを追加
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # マスクを取得
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(Image.fromarray(image))
            mask = self.transform(Image.fromarray(mask))  # マスクにもtransformを適用

        return image, mask  # 画像とマスクのペアを返す

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
def segment_with_sam(image):
    mask = mask_generator.generate(image)
    mask = mask_unite(image, mask)
    mask = mask.astype(np.uint8)

    return mask

# 画像の前処理
def preprocess_image(image):
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    return transform(image)

# オートエンコーダの学習
def train_autoencoder(model, train_loader, val_loader, num_epochs, gpu_ids):
    # 指定されたGPUを使用
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using GPUs: {gpu_ids}")

    model = DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for image, mask in train_loader:  # マスクも受け取る
            image = image.to(device)
            mask = mask.to(device)  # マスクをデバイスに移動

            optimizer.zero_grad()
            output = model(image)

            # 損失計算 (マスクを使用)
            loss_mse = criterion_mse(output, mask)
            loss = loss_mse

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for image, mask in val_loader:  # マスクも受け取る
                image = image.to(device)
                mask = mask.to(device)  # マスクをデバイスに移動

                output = model(image)

                # 損失計算 (マスクを使用)
                loss_mse = criterion_mse(output, mask)
                loss = loss_mse
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.module.state_dict(), 'best_model_sammask_simple_1024.pth')
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    return model.module

def get_files_in_directory(directory_path):
    # 指定されたディレクトリ内のファイル名を格納するリストを初期化
    files = []
    
    count = 0
    # ディレクトリ内のすべてのファイルとフォルダをリストアップ
    for item in os.listdir(directory_path):
        # アイテムへのフルパスを取得
        item_path = os.path.join(directory_path, item)
        if item == "36.jpg" or item == "42.jpg":
            continue
        
        # アイテムがファイルであれば、リストに追加
        if os.path.isfile(item_path):
            files.append(item_path)

            count += 1
    
    return files


# メイン処理
if __name__ == "__main__":
    gpu_ids = [5, 6, 7]
    # メインのGPUを設定
    torch.cuda.set_device(gpu_ids[0])

    # データセットとデータローダーの準備
    transform = transforms.Compose([
         transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    directory_path = "/mnt/data/datasets/SEM/dataset_v1/normal/"
    sem_datasets = get_files_in_directory(directory_path)

    # 学習前にすべての画像に対してマスクを実施
    masks = []
    for i, image_path in enumerate(sem_datasets):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = segment_with_sam(image)

        masked_img = image.copy()
        masked_img[mask == 255] = 255

        # 出力ファイル名
        output_filename = f"{image_path[len(directory_path):]}.png"
        output_path = os.path.join("output_normal_sam", output_filename)

        # 画像を保存
        cv2.imwrite(output_path, masked_img)
        masks.append(mask)


    # データセットを作成 (マスクを含む)
    full_dataset = AnomalyDataset(sem_datasets, masks, transform=transform)

    # データセットを訓練用と検証用に分割（8:2）
    train_rate = 0.8
    train_size = int(train_rate * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # オートエンコーダモデルの初期化と学習
    model = SimplerCNN()
    trained_model = train_autoencoder(model, train_loader, val_loader, num_epochs=100, gpu_ids=gpu_ids)
