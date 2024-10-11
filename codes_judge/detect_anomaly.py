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
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth").to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model=sam,points_per_side = 96, pred_iou_thresh=0.92)

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

## 単純なCNNモデルの定義
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),  # 1チャンネル出力 (マスク)
            nn.Sigmoid()  # 出力を0-1の範囲に
        )

    def forward(self, x):
        return self.features(x)

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
# テスト用のデータセット
class TestDataset(Dataset):
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

        return image, image_path  # 画像とパスを返す

def filter_noise(filtered_output, filtered_mask):

    # 円に近いノイズを検出するための処理
    contours, _ = cv2.findContours(filtered_output.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 輪郭の円形度を計算
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0: #ゼロ除算エラー回避
            circularity = 0
        else:
            circularity = 4 * np.pi * area / (perimeter * perimeter)

        # 円形度が高いほど、より大きなカーネルで膨張処理
        kernel_size = int(circularity * 8) + 1  # kernel_sizeを1以上に
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 個々の輪郭を膨張
        mask_contour = np.zeros_like(filtered_output)
        cv2.drawContours(mask_contour, [contour], -1, 1, thickness=cv2.FILLED)
        dilated_contour = cv2.dilate(mask_contour.astype(np.uint8), kernel, iterations=1)
        filtered_output = np.maximum(filtered_output, dilated_contour) # 元のfiltered_outputとマージ


    anomaly_map_ae = ((filtered_mask - filtered_output) > 0).astype(float)

    return anomaly_map_ae

# テストの実行
def test_autoencoder(model, test_loader, output_dir, gpu_ids):
    # 指定されたGPUを使用
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using GPUs: {gpu_ids}")

    model = DataParallel(model, device_ids=gpu_ids)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for image, image_path in test_loader:
            image = image.to(device)
            output = model(image)
            if "normal/36.jpg" in image_path or "normal/42.jpg" in image_path:
                continue

            # 出力をSAMに入力
            for i in range(output.size(0)):
                filtered_output = (output[i].mean(dim=0) > 0.75).float()

                image_numpy = (image[i].detach().cpu().numpy() * 255).astype(np.uint8)
                image_numpy = np.transpose(image_numpy, (1, 2, 0))  # (B, C, H, W) -> (B, H, W, C)
                mask_numpy = segment_with_sam(image_numpy)
                mask = preprocess_image(mask_numpy).to(device)

                # しきい値で物体判定
                # output(オートエンコーダの出力)は、黒い部分を物体として学習したので、反転する
                filtered_mask = (mask.mean(dim=0) > 0.9).float()

                # filtered_output = filtered_output.cpu().detach().numpy()
                # kernel = np.ones((4, 4), np.uint8)  # 構造要素 (5x5の正方形)
                # filtered_output = cv2.dilate(filtered_output, kernel, iterations=1)
                # filtered_mask = filtered_mask.cpu().detach().numpy()
                # anomaly_map_ae = ((filtered_mask - filtered_output) > 0).astype(float)
                filtered_output = filtered_output.cpu().detach().numpy()
                filtered_mask = filtered_mask.cpu().detach().numpy()
                anomaly_map_ae = filter_noise(filtered_output, filtered_mask)

                # 異常検出画像を出力
                plt.imshow(anomaly_map_ae, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                file_name = os.path.basename(image_path[i])
                output_filename = f"./data/anomaly/n2_{file_name}"
                plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
                print("Saved as", output_filename)

# メイン処理
if __name__ == "__main__":
    gpu_ids = [5, 6, 7]
    # メインのGPUを設定
    torch.cuda.set_device(gpu_ids[0])

    # モデルの読み込み
    model = SimplerCNN()
    model.load_state_dict(torch.load('../codes/best_model_sammask_simple_512.pth'))

    # テストデータの準備
    test_dir = "/mnt/data/datasets/SEM/dataset_v1/anomaly/n2"
    test_image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    test_dataset = TestDataset(test_image_paths, transform=transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ]))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 出力ディレクトリの作成
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # テストの実行
    test_autoencoder(model, test_loader, output_dir, gpu_ids)