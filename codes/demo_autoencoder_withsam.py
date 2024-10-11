import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2

# SAMモデルの初期化
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth")
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


# 画像の前処理
def preprocess_image(image):
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

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

# 異物検知
def detect_anomaly(model, image_path, threshold, device):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = segment_with_sam(image)
    
    image = preprocess_image(image).to(device)
    mask = preprocess_image(mask).to(device)
    
    input_tensor = torch.cat([image, mask], dim=1)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    diff = torch.abs(output - input_tensor)
    anomaly_map = diff.mean(dim=1).squeeze().cpu().numpy()
    
    return output.squeeze().cpu().numpy()

# メイン処理
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの読み込み
    model = AutoEncoder()
    model.load_state_dict(torch.load('autoencoder_model40.pth'))
    model.to(device)
    
    for count in range(100):
        # 異物検知の実行
        image_path = f"/mnt/data/datasets/SEM/dataset_v1/anomaly/n3/{count+1:02}.jpg"
        threshold = 0.1  # 適切なしきい値を設定
        anomaly_map = detect_anatomy_map = detect_anomaly(model, image_path, threshold, device)

        # 結果の表示
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 3, 1)
        plt.imshow(Image.open(image_path))
        plt.title("Original Image")
        plt.axis('off')
        
        for index, ann in enumerate(anomaly_map):
            plt.subplot(3, 3, index+2)
            plt.imshow(ann, cmap='hot')
            plt.title("Anomaly Detection Result")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        output_filename = f"output_anomaly/output_{count+1:02}.jpg"
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)