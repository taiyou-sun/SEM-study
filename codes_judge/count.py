import cv2
import numpy as np

def count_objects(image_path, min_area):
    """
    一定以上の面積を持つオブジェクトの数を数えます。

    Args:
        image_path: 画像ファイルのパス。
        min_area: オブジェクトとしてカウントする最小面積。

    Returns:
        オブジェクトの数。
    """

    # 画像を読み込み、グレースケールに変換
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")


    # 2値化 (0と1の画像なので、念のため)
    _, binary_img = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY)

    # オブジェクトの輪郭を検出
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 一定以上の面積を持つオブジェクトの数をカウント
    object_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            object_count += 1

    return object_count


# 使用例
image_path = "/home/omichi/segment-anything/codes_judge/data/train/anomaly/01.jpg"  # 画像ファイルのパスを指定
min_area = 200  # 最小面積を指定

try:
    num_objects = count_objects(image_path, min_area)
    print(f"面積が{min_area}以上のオブジェクトの数: {num_objects}")

except FileNotFoundError as e:
    print(e)


# デバッグ用に、検出されたオブジェクトを描画する例 (必要に応じてコメントアウトを外してください)
"""
import matplotlib.pyplot as plt

# ... (count_objects 関数など) ...

# 検出されたオブジェクトを描画
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # カラー画像に変換
for contour in contours:
    area = cv2.contourArea(contour)
    if area >= min_area:
        cv2.drawContours(img_rgb, [contour], -1, (0, 255, 0), 2) # 緑色で輪郭を描画

plt.imshow(img_rgb)
plt.title(f"Detected Objects (Area >= {min_area})")
plt.show()
"""
