#
"""
オートエンコーダの学習の際、SAMを使わず異常のないSEM画像だけ使って学習を行い，
SAMのセグメンテーション結果から学習モデルの出力のノイズ除去をする
train_autoencoder_nosam.pyからpthファイルを得る
"""
#
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 学習スクリプトからモデルと関数をインポート
from train_autoencoder_nosam import AutoEncoder

# SAMモデルの初期化
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
DEVICE = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth").to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model=sam)

# 画像の前処理
def preprocess_image(image):
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
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

# 異常検知（SAMとオートエンコーダの結果を組み合わせる）
def detect_anomaly(model, image_path, mask_thres, output_thres, device):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = segment_with_sam(image)
    
    image = preprocess_image(image).to(device)
    mask = preprocess_image(mask).to(device)
    
    # オートエンコーダの結果
    with torch.no_grad():
        output = model(image)

    # しきい値で物体判定
    # output(オートエンコーダの出力)は、黒い部分を物体として学習したので、反転する
    filted_mask = (mask.mean(dim=1) > mask_thres).float()
    filted_output = ((1 - output.mean(dim=1)) > output_thres).float()
    
    diff = ((filted_mask - filted_output) > 0).float()
    anomaly_map_ae = diff.squeeze().cpu().numpy()
    
    return anomaly_map_ae, filted_mask.squeeze().cpu().numpy(), filted_output.squeeze().cpu().numpy()
# メイン処理
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの読み込み
    model = AutoEncoder()
    model.load_state_dict(torch.load('autoencoder_model_nosam50.pth'))
    model.to(device)
    
    for count in range(100):
        # 異物検知の実行
        image_path = f"/mnt/data/datasets/SEM/dataset_v1/anomaly/n3/{count+1:02}.jpg"
        # 適切なしきい値を設定
        mask_thres = 0.05  
        output_thres = 0.5
        anomaly_map,filted_mask, filted_output = detect_anatomy_map = detect_anomaly(model, image_path, mask_thres,output_thres, device)

        # 結果の表示
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(Image.open(image_path))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(anomaly_map, cmap='hot')
        plt.title("Anomaly Detection Result")
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(filted_mask, cmap='hot')
        plt.title("SAM Mask Detection Result")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(filted_output, cmap='hot')
        plt.title("AutoEncoder Detection Result")
        plt.axis('off')
    
        plt.tight_layout()
        plt.show()
        output_filename = f"output_anomaly/output_{count+1:02}.jpg"
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        print("Saved as", output_filename)