import os
import torch
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2

# SAMモデルの初期化
DEVICE = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint="/home/omichi/segment-anything/sam_vit_h_4b8939.pth").to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model=sam)

def get_files_in_directory(directory_path, max_amount):
    # 指定されたディレクトリ内のファイル名を格納するリストを初期化
    files = []
    
    # ディレクトリ内のすべてのファイルとフォルダをリストアップ
    for idx, item in enumerate(os.listdir(directory_path)):
        if idx > max_amount:
            break
        # アイテムへのフルパスを取得
        item_path = os.path.join(directory_path, item)
        
        # アイテムがファイルであれば、リストに追加
        if os.path.isfile(item_path):
            files.append(item_path)
    
    return files

def mask_unite(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)
        img[m] = color_mask
    return img

def generate_sam_masks(img_paths):
    for idx, img_path in enumerate(img_paths):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        masks = mask_unite(masks)
        plt.imshow(masks)
        plt.axis('off')
        output_filename = f"../filtered_sam/{idx}.jpg"
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        print(f"{img_path} is done({idx+1}/{len(img_paths)}).")

directory_path = "/mnt/data/datasets/SEM/dataset_v1/normal/"
sem_paths = get_files_in_directory(directory_path, 80)
generate_sam_masks(sem_paths)