#
"""
オートエンコーダの出力で，SAMのセグメンテーションの結果を出力させる
入力が画像そのまま，出力がSAMのセグメンテーション画像

学習ではオートエンコーダでも，よりモデルの複雑なものを使う
"""
#
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 学習スクリプトからモデルと関数をインポート
from train_autoencoder_withsam3_v2 import ImprovedAutoEncoder

# SAMモデルの初期化
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
DEVICE = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth").to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model=sam, pred_iou_thresh=0.90)

# 画像の前処理
def preprocess_image(image):
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

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
def segment_with_sam(image):
    mask = mask_generator.generate(image)
    mask = mask_unite(image, mask)
    mask = mask.astype(np.uint8)
    return mask

# 異常検知（SAMとオートエンコーダの結果を組み合わせる）
def detect_anomaly(model, image_path, mask_thres, output_thres, device):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = preprocess_image(image).to(device)
    
    # オートエンコーダの結果
    with torch.no_grad():
        output = model(image)

        image_numpy = (image.detach().cpu().numpy() * 255).astype(np.uint8)
        image_numpy = np.transpose(image_numpy, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
        mask_numpy = segment_with_sam(image_numpy[0])
        mask = preprocess_image(mask_numpy).to(device)

        # しきい値で物体判定
        # output(オートエンコーダの出力)は、黒い部分を物体として学習したので、反転する
        filted_mask = (mask.mean(dim=1) > mask_thres).float()
        filted_output = (output.mean(dim=1) > output_thres).float()
        
        diff = ((filted_mask - filted_output) > 0).float()
        anomaly_map_ae = diff.squeeze().cpu().numpy()

        output_for_sam = (output.detach().cpu().numpy() * 255).astype(np.uint8)
        output_for_sam = np.transpose(output_for_sam, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
    
        return anomaly_map_ae, filted_mask.squeeze().cpu().numpy(), filted_output.squeeze().cpu().numpy()
# メイン処理
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの読み込み
    model = ImprovedAutoEncoder()
    state_dict = torch.load('best_model_withsam3_v2_2.pth')
    # キーから'module.'を削除
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    for count in range(100):
        # 異物検知の実行
        image_path = f"/mnt/data/datasets/SEM/dataset_v1/anomaly/n3/{count+1:02}.jpg"
        # 適切なしきい値を設定
        mask_thres = 0.9
        output_thres = 0.6
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
        output_filename = f"output_test/output_{count+1:02}.jpg"
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        print("Saved as", output_filename)