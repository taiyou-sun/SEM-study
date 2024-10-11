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
mask_generator = SamAutomaticMaskGenerator(model=sam, pred_iou_thresh=0.90)

# オートエンコーダモデルの定義（入力チャンネル数を3に変更）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ImprovedAutoEncoder(nn.Module):
    def __init__(self):
        super(ImprovedAutoEncoder, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(6)])
        
        # Decoder
        self.dec4 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(512, 128)
        self.dec2 = self.upconv_block(256, 64)
        self.dec1 = self.upconv_block(128, 32)
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Residual blocks
        res = self.res_blocks(e4)
        
        # Decoder with skip connections
        d4 = self.dec4(res)
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        
        return self.final(d1)

# 知覚損失のための VGG モデル
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

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
        
        if self.transform:
            image = self.transform(Image.fromarray(image))
        
        return image

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
    criterion_perceptual = VGGPerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for image in train_loader:
            image = image.to(device)

            optimizer.zero_grad()
            output = model(image)

            images_numpy = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            images_numpy = np.transpose(images_numpy, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
            masks_numpy = []
            for image_numpy in images_numpy:
                mask_numpy = segment_with_sam(image_numpy)
                mask_numpy = preprocess_image(mask_numpy).to(device)
                masks_numpy.append(mask_numpy)
            mask = torch.stack(masks_numpy)

            mask_for_sam = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
            mask_for_sam = np.transpose(mask_for_sam, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
            plt.figure(figsize=(12, 6))
            plt.imshow(mask_for_sam[0], cmap='hot')
            plt.axis('off')
            mask_filename = f"output_anomaly2/sam3mask_{epoch:02}.jpg"
            plt.savefig(mask_filename, bbox_inches='tight', pad_inches=0)
            
            output_for_sam = (output.detach().cpu().numpy() * 255).astype(np.uint8)
            output_for_sam = np.transpose(output_for_sam, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
            plt.figure(figsize=(12, 6))
            plt.imshow(output_for_sam[0], cmap='hot')
            plt.axis('off')
            output_filename = f"output_anomaly2/sam3output_{epoch:02}.jpg"
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)

            loss_mse = criterion_mse(output, mask)
            loss_perceptual = sum(criterion_mse(o, t) for o, t in zip(criterion_perceptual(output), criterion_perceptual(mask)))
            loss = loss_mse + 0.1 * loss_perceptual

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for image in val_loader:
                image = image.to(device)
                output = model(image)

                images_numpy = (image.detach().cpu().numpy() * 255).astype(np.uint8)
                images_numpy = np.transpose(images_numpy, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
                masks_numpy = []
                for image_numpy in images_numpy:
                    mask_numpy = segment_with_sam(image_numpy)
                    mask_numpy = preprocess_image(mask_numpy).to(device)
                    masks_numpy.append(mask_numpy)
                mask = torch.stack(masks_numpy)

                loss_mse = criterion_mse(output, mask)
                loss_perceptual = sum(criterion_mse(o, t) for o, t in zip(criterion_perceptual(output), criterion_perceptual(mask)))
                loss = loss_mse + 0.1 * loss_perceptual
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.module.state_dict(), 'best_model_withsam3_v2_2.pth')
        
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
    full_dataset = AnomalyDataset(sem_datasets, transform=transform)

    # データセットを訓練用と検証用に分割（8:2）
    train_rate = 0.9
    train_size = int(train_rate * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # オートエンコーダモデルの初期化と学習
    model = ImprovedAutoEncoder()
    trained_model = train_autoencoder(model, train_loader, val_loader, num_epochs=100, gpu_ids=gpu_ids)
