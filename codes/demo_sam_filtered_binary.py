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

def remove_black_dots(image_path, threshold, min_area=50):
    # 画像の読み込み
    image = cv2.imread(image_path, 0)  # グレースケールで読み込み    

    # 手動でしきい値を設定して二値化
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 輪郭検出
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 小さな黒い点（輪郭）を除去
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            cv2.drawContours(opening, [contour], 0, 255, -1)

    return opening

# 異常検知（SAMとオートエンコーダの結果を組み合わせる）
def detect_anomaly(image_path, mask_thres, device):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = preprocess_image(image).to(device)
    filtered_image = preprocess_image(remove_black_dots(image_path, threshold=70, min_area=50)).to(device).squeeze(1)

    image_numpy = (image.detach().cpu().numpy() * 255).astype(np.uint8)
    image_numpy = np.transpose(image_numpy, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
    mask_numpy = segment_with_sam(image_numpy[0])
    mask = preprocess_image(mask_numpy).to(device)

    # しきい値で物体判定
    # output(オートエンコーダの出力)は、黒い部分を物体として学習したので、反転する
    filted_mask = (mask.mean(dim=1) > mask_thres).float()
    print(filted_mask.shape)
    print(filtered_image.shape)
    output = filted_mask * filtered_image

    return output.squeeze().cpu().numpy(), filted_mask.squeeze().cpu().numpy(), filtered_image.squeeze().cpu().numpy()
# メイン処理
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for count in range(100):
        # 異物検知の実行
        image_path = f"/mnt/data/datasets/SEM/dataset_v1/anomaly/n3/{count+1:02}.jpg"
        # 適切なしきい値を設定
        mask_thres = 0.9
        output_thres = 0.6
        anomaly_map,filted_mask, filted_output = detect_anatomy_map = detect_anomaly(image_path, mask_thres, device)

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
        plt.title("Binary Detection Result")
        plt.axis('off')
    
        plt.tight_layout()
        plt.show()
        output_filename = f"output_test/output_{count+1:02}.jpg"
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        print("Saved as", output_filename)