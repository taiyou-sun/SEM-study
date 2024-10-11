from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# モデルのロード
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")

# マスク生成器の作成
mask_generator = SamAutomaticMaskGenerator(model=sam)

# 画像の読み込み
filename = "/home/omichi/segment-anything/demo/src/assets/data/anomaly_01.jpg"  # 画像のパスを指定
image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# マスクの生成
masks = mask_generator.generate(image)

# バウンディングボックスの最小面積設定
min_area = 3000  # 最小面積を指定 (ピクセル単位)

# バウンディングボックスの抽出と画像への描画
fig, ax = plt.subplots(1)
ax.imshow(image)

for mask in masks:
    bbox = mask['bbox']
    x, y, w, h = bbox
    area = w * h
    if area >= min_area:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
plt.axis('off')
plt.show()
# 結果を保存する場合
output_filename = "output_with_bboxes.jpg"
fig.savefig(output_filename, bbox_inches='tight', pad_inches=0)