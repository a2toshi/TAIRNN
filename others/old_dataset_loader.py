import os
import sys

import numpy as np
import torch
from PIL import Image


# Z-score正規化
def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


def dataset_loader(folder_names, root_dir, joint_dim=3, im_size=64, mean=None, std=None):
    # 画像サイズ指定をタプル化
    if isinstance(im_size, int):
        target_size = (im_size, im_size)
    else:
        target_size = tuple(im_size)

    # データを収集
    all_images = []
    max_images_count = 0
    all_positions = []

    for folder in folder_names:
        folder_path = os.path.join(root_dir, folder)

        # "image"で始まるファイルを収集
        image_files = sorted([file for file in os.listdir(folder_path) if file.startswith("image")])

        # 画像をリストに読み込む
        images = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert("RGB")
            # img = img.resize(target_size, resample=Image.BILINEAR)
            img = img.resize(target_size)
            img = np.array(img)
            img = img / 255.0  # 正規化
            images.append(img)

        # 画像のリストを配列に変換
        images = np.array(images)
        max_images_count = max(max_images_count, images.shape[0])
        all_images.append(images)

        # 右鉗子の位置情報
        position_files = sorted([file for file in os.listdir(folder_path) if file.startswith("position")])
        folder_positions = []

        for pos_file in position_files:
            pos_path = os.path.join(folder_path, pos_file)

            # ファイルを読み込んで、最初(右鉗子)のjoint_dim個の数字を抽出
            with open(pos_path, "r") as f:
                data = f.read().strip().split(",")
                first_three = [float(data[i]) for i in range(joint_dim)]

            # 位置情報を保存
            folder_positions.append(first_three)
        # 各フォルダの位置情報を保存
        all_positions.append(np.array(folder_positions))

    # 各フォルダの画像数が異なる場合、少ないフォルダにはパディングを追加
    padded_images = []
    for images in all_images:
        padding_needed = max_images_count - images.shape[0]
        if padding_needed > 0:
            last_image = images[-1]  # 最後の画像を取得
            padding_images = np.tile(last_image, (padding_needed, 1, 1, 1))
            images = np.vstack([images, padding_images])
        padded_images.append(images)
    all_images_array = np.stack(padded_images, axis=0)

    # 位置情報の正規化とパディング
    normalized_all_positions = []
    for positions in all_positions:
        if mean is None and std is None:
            normalized_positions, current_mean, current_std = z_score_normalize(positions)
            mean = current_mean if mean is None else mean
            std = current_std if std is None else std
        else:
            normalized_positions = (positions - mean) / std
        normalized_all_positions.append(normalized_positions)

    max_length = max(position.shape[0] for position in all_positions)
    padded_normalized_positions = []
    for positions in normalized_all_positions:
        if len(positions) < max_length:
            last_value = positions[-1]  # 最後の位置情報
            padding = np.tile(last_value, (max_length - len(positions), 1))
            padded_positions = np.vstack([positions, padding])
        else:
            padded_positions = positions
        padded_normalized_positions.append(padded_positions)

    all_positions_array = np.array(padded_normalized_positions)
    all_images_array = np.transpose(all_images_array, (0, 1, 4, 2, 3))

    # 不要な変数のメモリを解放
    del padded_images
    del all_images
    del all_positions
    del padded_positions
    del images
    torch.cuda.empty_cache()

    return all_images_array, all_positions_array, mean, std
