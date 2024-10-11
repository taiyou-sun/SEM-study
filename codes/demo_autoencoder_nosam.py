import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 学習スクリプトからモデルと関数をインポート
from train_autoencoder_nosam import AutoEncoder

# SAMモデルの初期化
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# 画像の前処理
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# 異常検知（SAMとオートエンコーダの結果を組み合わせる）
def detect_anomaly(model, image_path, threshold, device):
    model.eval()
    image = preprocess_image(image_path).to(device)
    
    # オートエンコーダの結果
    with torch.no_grad():
        output = model(image)
    
    diff = torch.abs(image - output)
    anomaly_map_ae = diff.mean(dim=1).squeeze().cpu().numpy()
    
    # SAMの結果
    #mask = segment_with_sam(image).to(device)
    
    # SAMの結果とオートエンコーダの結果を組み合わせる
    #anomaly_map_ae = mask.squeeze().cpu().numpy()
    
    return anomaly_map_ae
# メイン処理
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの読み込み
    model = AutoEncoder()
    model.load_state_dict(torch.load('../autoencoder_model_nosam.pth'))
    model.to(device)
    
    for count in range(100):
        # 異物検知の実行
        image_path = f"/mnt/data/datasets/SEM/dataset_v1/anomaly/n3/{count+1:02}.jpg"
        threshold = 0.1  # 適切なしきい値を設定
        anomaly_map = detect_anatomy_map = detect_anomaly(model, image_path, threshold, device)

        print(anomaly_map)
        # 結果の表示
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(image_path))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(anomaly_map, cmap='hot')
        plt.title("Anomaly Detection Result")
        plt.axis('off')
            
        
        plt.tight_layout()
        plt.show()
        output_filename = f"output_anomaly/output_{count+1:02}.jpg"
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)