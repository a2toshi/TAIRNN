import glob
import os

import imageio  # このライブラリがない場合は pip install imageio[ffmpeg] でインストールしてください
import natsort  # このライブラリがない場合は pip install natsort でインストールしてください
from PIL import Image


def create_gif_from_image_sequence(image_pattern, output_path, fps=5):
    """
    指定されたファイルパターンに一致する画像群からGIFアニメーションを生成する。

    Args:
        image_pattern (str): 画像ファイルを検索するためのglobパターン (例: "path/to/step*_kv_features.png")
        output_path (str): 生成されるGIFの保存先パス。
        fps (int): GIFのフレームレート（1秒あたりのフレーム数）。
    """
    print(f"Searching for images with pattern: {image_pattern}")

    # natsortを使ってファイル名を自然順（step0, step1, step2...）にソート
    file_paths = natsort.natsorted(glob.glob(image_pattern))

    if not file_paths:
        print(f"Warning: No images found for pattern '{os.path.basename(image_pattern)}'. Skipping.")
        return

    print(f"Found {len(file_paths)} images. Creating GIF...")

    # 画像を読み込んでリストに追加
    try:
        frames = [Image.open(file) for file in file_paths]
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # GIFを生成して保存
    try:
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Successfully saved GIF to: {output_path}")
    except Exception as e:
        print(f"Error creating GIF: {e}")


if __name__ == "__main__":
    # --- ここを修正 ---
    # 実際にPNG画像が保存されているディレクトリのパスを指定してください
    log_directory = "/work/gn45/n45002/TAiRNN/TAiRNN/output/20250709_1800_00/20250621_153730"

    # 生成するGIFのフレームレート（1秒あたりのフレーム数）
    frames_per_second = 10

    # --- 生成する3種類のGIFの設定 ---
    gif_tasks = {
        "kv_features": "step*_kv_features.png",
        "input_attn": "step*_input_attn_overlay.png",
        "pred_heatmap": "step*_pred_heatmap_overlay.png",
    }

    # 各タスクを実行
    for task_name, file_pattern in gif_tasks.items():
        # 入力画像の検索パターンを作成
        image_pattern = os.path.join(log_directory, file_pattern)

        # 出力ファイル名を決定
        output_filename = f"{task_name}_animation.gif"
        output_path = os.path.join(log_directory, output_filename)

        # GIF生成関数を呼び出し
        create_gif_from_image_sequence(image_pattern, output_path, fps=frames_per_second)
        print("-" * 30)
