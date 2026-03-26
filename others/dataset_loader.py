import os

import numpy as np
from PIL import Image


def min_max_normalize(data, data_min=None, data_max=None):
    """
    データセット全体にわたる単一の最小値・最大値を用いて正規化を行う。
    """
    if data_min is None or data_max is None:
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)

    # ゼロ除算を避けるための小さな値を分母に加える
    range_ = data_max - data_min
    range_[range_ == 0] = 1e-9

    normalized_data = (data - data_min) / range_
    return normalized_data, data_min, data_max


def dataset_loader(folder_names, root_dir, data_min=None, data_max=None, joint_dim=3, img_size=64):
    all_images = []
    all_positions = []
    crop_box = (160, 0, 1440, 1000)  # クロッピング領域
    dec = 2

    # 1. まず全フォルダから画像と位置情報をメモリにロード
    for folder in folder_names:
        folder_path = os.path.join(root_dir, folder)
        if folder.startswith("."):
            continue

        print(f"Loading data from: {folder}")

        # 画像ファイルの収集と処理
        image_files = sorted([f for f in os.listdir(folder_path) if f.startswith("image")])
        images = []
        cnt = -1
        for img_file in image_files:
            cnt += 1
            if cnt % dec != 0:
                continue
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert("RGB")
            img = img.crop(crop_box)
            img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)  # 高品質なリサイズ
            img_np = np.asarray(img, dtype=np.float32) / 255.0
            images.append(img_np)

        if images:
            all_images.append(np.array(images))

        # 位置情報ファイルの収集と処理
        position_files = sorted([f for f in os.listdir(folder_path) if f.startswith("position")])
        folder_positions = []
        cnt = -1
        for pos_file in position_files:
            cnt += 1
            if cnt % dec != 0:
                continue
            pos_path = os.path.join(folder_path, pos_file)
            with open(pos_path, "r") as f:
                data = f.read().strip().split(",")
                # 変更
                selected = [float(data[i]) for i in range(joint_dim)] + [float(data[i]) for i in range(9, 12)]
                folder_positions.append(selected)

        if folder_positions:
            all_positions.append(np.array(folder_positions))

    # 2. 位置情報の正規化
    # まず全時系列データを結合して、データセット全体の最小値・最大値を計算
    if data_min is None or data_max is None:
        print("Calculating min/max values from the entire dataset...")
        combined_positions = np.vstack(all_positions)
        _, data_min, data_max = min_max_normalize(combined_positions)
        is_train_run = True  # これは学習データでの実行であることを示す
    else:
        # テスト時には学習時のmin/maxが渡される
        is_train_run = False

    normalized_all_positions = [min_max_normalize(p, data_min, data_max)[0] for p in all_positions]

    # 3. 画像と位置情報のパディング（シーケンス長を揃える）
    max_len = max(len(p) for p in normalized_all_positions)

    padded_images = []
    for images in all_images:
        pad_needed = max_len - len(images)
        if pad_needed > 0:
            padding = np.tile(images[-1], (pad_needed, 1, 1, 1))
            images = np.concatenate([images, padding], axis=0)
        padded_images.append(images)

    padded_positions = []
    for positions in normalized_all_positions:
        pad_needed = max_len - len(positions)
        if pad_needed > 0:
            padding = np.tile(positions[-1], (pad_needed, 1))
            positions = np.concatenate([positions, padding], axis=0)
        padded_positions.append(positions)

    # 最終的なNumPy配列に変換
    all_images_array = np.array(padded_images)
    all_images_array = np.transpose(all_images_array, (0, 1, 4, 2, 3))
    all_positions_array = np.array(padded_positions)

    # 学習実行時のみmin/maxを返す。テスト時はNoneを返す
    if is_train_run:
        return all_images_array, all_positions_array, data_min, data_max
    else:
        return all_images_array, all_positions_array, None, None


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    root_dir = "test_20241116_ex1"
    folder_names = sorted([name for name in os.listdir(root_dir)])
    #     print(os.listdir(root_dir))
    folder_names = folder_names[:3]
    #     print(folder_names)
    all_images_array, all_positions_array, mean, std = dataset_loader(folder_names, root_dir)

    # 画像データの形状を確認
    print("All images shape:", all_images_array.shape)  # (フォルダ数, 画像数, チャネル数, 高さ, 幅)

    # 特定のフォルダと画像のインデックスを指定して表示
    folder_idx = 0  # 表示するフォルダのインデックス
    image_idx = 0  # 表示する画像のインデックス

    # 指定した画像を取得 (チャネル順が (C, H, W) → (H, W, C) に変換)
    image_to_show = all_images_array[folder_idx, image_idx].transpose(1, 2, 0)

    # 画像を表示
    plt.imshow(image_to_show)
    plt.title(f"Folder {folder_idx}, Image {image_idx}")
    plt.axis("off")  # 軸を非表示
    plt.show()
