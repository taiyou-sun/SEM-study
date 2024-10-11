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
mask_generator = SamAutomaticMaskGenerator(model=sam,points_per_side = 64, pred_iou_thresh=0.90)

# 単純なU-Netモデルの定義
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Decoder
        self.dec4 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(256 + 256, 128)
        self.dec2 = self.upconv_block(128 + 128, 64)
        self.dec1 = self.upconv_block(64 + 64, 3)

        self.final = nn.Sigmoid()  # 出力を0-1の範囲に

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder
        d4 = self.dec4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)

        return self.final(d1)

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
         transforms.Resize((512, 512)),
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
            torch.save(model.module.state_dict(), 'best_model_sammask.pth')
        
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
         transforms.Resize((512, 512)),
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
    model = SimpleUNet()
    trained_model = train_autoencoder(model, train_loader, val_loader, num_epochs=100, gpu_ids=gpu_ids)