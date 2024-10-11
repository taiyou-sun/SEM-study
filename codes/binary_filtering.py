import cv2
import numpy as np

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

    # 結果を保存
    cv2.imwrite('result.png', opening)
    cv2.imwrite('result2.png', image)

    print("処理が完了しました。結果はresult.pngに保存されています。")

# 使用例
remove_black_dots('/mnt/data/datasets/SEM/dataset_v1/anomaly/n3/01.jpg', threshold=100, min_area=50)
