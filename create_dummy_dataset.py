"""
ダミーデータセット生成スクリプト
train_TAiRNN_conv_wo.py の動作確認用に最小限のデータを生成する。

生成されるデータ:
  dummy_dataset/
    episode_001/ ... episode_004/
      image_000000.png ... image_000049.png  (50フレーム)
      position_000000.txt ... position_000049.txt
"""

import os
import numpy as np
from PIL import Image

OUTPUT_DIR = "dummy_dataset"
N_EPISODES = 4   # train_test_split が動くための最低エピソード数
N_FRAMES = 50    # chunk_size(30) より多くする

rng = np.random.default_rng(seed=0)


def make_position_line():
    """
    12列以上のカンマ区切り数値を1行で返す。
    使用される列: 0,1,2 (EEF XYZ mm), 7 (gripper 0-1), 9,10,11 (cam XYZ mm)
    """
    vals = rng.uniform(-500, 500, size=12).tolist()
    vals[7] = rng.uniform(0.0, 1.0)     # gripper: 0-1
    return ",".join(f"{v:.6f}" for v in vals)


def make_image(size=256):
    """ランダムなRGB画像を返す (元解像度はなんでもよい)"""
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ep in range(1, N_EPISODES + 1):
        ep_dir = os.path.join(OUTPUT_DIR, f"episode_{ep:03d}")
        os.makedirs(ep_dir, exist_ok=True)

        for t in range(N_FRAMES):
            img = make_image()
            img.save(os.path.join(ep_dir, f"image_{t:06d}.png"))

            pos_line = make_position_line()
            with open(os.path.join(ep_dir, f"position_{t:06d}.txt"), "w") as f:
                f.write(pos_line + "\n")

        print(f"Created {ep_dir}  ({N_FRAMES} frames)")

    print(f"\nDone. Dataset: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
