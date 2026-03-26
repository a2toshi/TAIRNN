import os

import numpy as np
from PIL import Image


def min_max_normalize(data, data_min=None, data_max=None):
    if data_min is None or data_max is None:
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)

    # ゼロ除算を避けるための小さな値を分母に加える
    range_ = data_max - data_min
    range_[range_ == 0] = 1e-9

    normalized_data = (data - data_min) / range_
    return normalized_data, data_min, data_max


def dataset_loader(
    folder_names, root_dir, data_min=None, data_max=None, joint_dim=3, img_size=64, chunk_size=30, decrement=1
):
    all_images = []
    all_positions = []
    crop_box = (160, 0, 1440, 1000)

    # 1. 各フォルダごとにシーケンスをロード（エピソード単位）
    for folder in folder_names:
        folder_path = os.path.join(root_dir, folder)
        if folder.startswith("."):
            continue

        print(f"Loading data from: {folder}")

        # 画像
        image_files = sorted([f for f in os.listdir(folder_path) if f.startswith("image")])
        images = []
        cnt = -1
        for img_file in image_files:
            cnt += 1
            if cnt % decrement != 0:
                continue
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert("RGB")
            img = img.crop(crop_box)
            img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            img_np = np.asarray(img, dtype=np.float32) / 255.0
            images.append(img_np)

        if images:
            all_images.append(np.array(images))

        # 位置
        position_files = sorted([f for f in os.listdir(folder_path) if f.startswith("position")])
        folder_positions = []
        cnt = -1
        for pos_file in position_files:
            cnt += 1
            if cnt % decrement != 0:
                continue
            pos_path = os.path.join(folder_path, pos_file)
            with open(pos_path, "r") as f:
                data = f.read().strip().split(",")
                selected = (
                    [float(data[i]) for i in range(3)] + [float(data[7])] + [float(data[i]) for i in range(9, 12)]
                )  # [9:12]はカメラ座標系
                folder_positions.append(selected)

        if folder_positions:
            all_positions.append(np.array(folder_positions))

    # 2. 位置情報の正規化
    if data_min is None or data_max is None:
        print("Calculating min/max values from the entire dataset...")
        combined_positions = np.vstack(all_positions)
        _, data_min, data_max = min_max_normalize(combined_positions)
        is_train_run = True  # これは学習データでの実行であることを示す
    else:
        # テスト時には学習時のmin/maxが渡される
        is_train_run = False

    normalized_all_positions = [min_max_normalize(p, data_min, data_max)[0] for p in all_positions]

    # 3. エピソードごとに chunk_size でチャンク化 + 各チャンク内だけ末尾パディング
    chunk_images = []  # list of (chunk_size, H, W, 3)
    chunk_positions = []  # list of (chunk_size, D)

    for imgs, pos in zip(all_images, normalized_all_positions):
        T = len(pos)

        start = 0
        while start < T:
            end = start + chunk_size
            if end <= T:
                img_chunk = imgs[start:end]
                pos_chunk = pos[start:end]
            else:
                # 末尾が足りない分だけ、そのエピソードの最後の状態でパディング
                valid_len = T - start  # 実データの長さ
                if valid_len <= 0:
                    break  # もう実データが無い
                pad_len = chunk_size - valid_len

                img_valid = imgs[start:T]  # (valid_len, H, W, 3)
                pos_valid = pos[start:T]  # (valid_len, D)

                last_img = imgs[T - 1 : T]  # (1, H, W, 3)
                last_pos = pos[T - 1 : T]  # (1, D)

                img_pad = np.repeat(last_img, pad_len, axis=0)
                pos_pad = np.repeat(last_pos, pad_len, axis=0)

                img_chunk = np.concatenate([img_valid, img_pad], axis=0)
                pos_chunk = np.concatenate([pos_valid, pos_pad], axis=0)

            chunk_images.append(img_chunk)
            chunk_positions.append(pos_chunk)

            start += chunk_size  # オーバーラップさせない場合

    # 4. 最終的な NumPy 配列に変換
    all_images_array = np.array(chunk_images)  # (N_chunks, L, H, W, 3)
    all_images_array = np.transpose(all_images_array, (0, 1, 4, 2, 3))  # -> (N_chunks, L, 3, H, W)

    all_positions_array = np.array(chunk_positions)  # (N_chunks, L, D)

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
