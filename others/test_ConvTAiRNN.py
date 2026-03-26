import argparse
import os

import matplotlib.animation as anim
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset_loader import dataset_loader

# model.pyからConvTAiRNNをインポート
from ModelCode.model import ConvTAiRNN
from ModelCode.utils import deprocess_img, restore_args, tensor2numpy

# --- 1. 引数とパラメータの読み込み (変更なし) ---
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True)
parser.add_argument("--idx", type=int, default=15)  # インデックスを整数型に
args = parser.parse_args()

model_folder = f"./log/{args.filename}/SARNN_latest.pth"
dir_name = os.path.dirname(model_folder)
params = restore_args(os.path.join(dir_name, "args.json"))

norm_params_path = os.path.join(dir_name, "norm_params.npz")
if not os.path.exists(norm_params_path):
    raise FileNotFoundError(f"norm_params.npz not found in {dir_name}.")
norm_data = np.load(norm_params_path)
data_min = norm_data["data_min"]
data_max = norm_data["data_max"]
print("Loaded normalization parameters successfully.")
print(data_max)

# --- 2. データセットの読み込み (修正あり) ---
root_dir = "/work/gn45/n45002/savedata/testdata_20250621_v2/"
all_test_folders = sorted([name for name in os.listdir(root_dir)])
test_dataset_name = all_test_folders[args.idx]
print(f"Testing on dataset: {test_dataset_name} (index: {args.idx})")

joint_dim = params["joint_dim"]

images, joints, _, _ = dataset_loader(
    [test_dataset_name], root_dir, data_min=data_min, data_max=data_max, joint_dim=joint_dim
)
images = images[0]
joints = joints[0]

true_joints = joints * (data_max - data_min) + data_min

img_size = params["im_size"]
model_name = args.filename

# --- 3. 出力フォルダとモデルの定義 (修正あり) ---
output_folder = os.path.join("TAiRNN/output", model_name, test_dataset_name)
print(f"Output will be saved to: {output_folder}")
os.makedirs(output_folder, exist_ok=True)  # GIF保存用にフォルダを先に作成

model = ConvTAiRNN(
    rec_dim=params["rec_dim"],
    joint_dim=params["joint_dim"],
    k_dim=params["k_dim"],
    temperature=params["temperature"],
    im_size=[img_size, img_size],
    visualize_on_forward=True,
    output_dir=output_folder,
)

ckpt = torch.load(model_folder, map_location=torch.device("cpu"), weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Model loaded successfully.")

# --- 4. 推論ループ (大幅な修正あり) ---
pred_image_list, pred_joint_list = [], []
input_attn_list, pred_heatmap_list = [], []
state = None

for i in range(len(images) - 1):
    img_t = torch.Tensor(np.expand_dims(images[i], 0))
    joint_t = torch.Tensor(np.expand_dims(joints[i], 0))

    # モデルの戻り値を新しい形式で受け取る
    y_image, y_joint, attn_map, pred_heatmap, state = model(img_t, joint_t, state)

    # 予測画像をリストに追加
    pred_img_np = tensor2numpy(y_image[0])
    pred_img_np = deprocess_img(pred_img_np, 0, 1).transpose(1, 2, 0)
    pred_image_list.append(pred_img_np)

    # 予測関節角を非正規化してリストに追加 (正しい式に修正)
    # # Dif Trainの場合
    # pred_joint_np = tensor2numpy(y_joint[0] + joint_t)
    pred_joint_np = tensor2numpy(y_joint[0])
    pred_joint_np = pred_joint_np * (data_max[:joint_dim] - data_min[:joint_dim]) + data_min[:joint_dim]
    pred_joint_list.append(pred_joint_np)

    # マップをリストに追加
    input_attn_list.append(tensor2numpy(attn_map[0]))
    pred_heatmap_list.append(tensor2numpy(pred_heatmap[0]))

    print(f"Processed step: {i+1}/{len(images)-1}")

pred_images = np.array(pred_image_list)
pred_joints = np.array(pred_joint_list)
input_attns = np.array(input_attn_list)
pred_heatmaps = np.array(pred_heatmap_list)

# --- 5. アニメーション生成 (全面的な見直し) ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=100)
T = len(pred_images)


def create_overlay(base_img, heatmap_k, cmap="jet", alpha=0.5):
    """単一のヒートマップをリサイズし、元画像に重ねる"""
    # PyTorchのinterpolateを使ってリサイズ
    heatmap_tensor = torch.from_numpy(heatmap_k).unsqueeze(0).unsqueeze(0)
    resized_map = F.interpolate(heatmap_tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    resized_map_np = resized_map.squeeze().numpy()

    # 重ね合わせ
    overlay_img = base_img.copy()
    colored_map = plt.get_cmap(cmap)(resized_map_np)[:, :, :3]
    overlay_img = (1 - alpha) * overlay_img + alpha * colored_map
    return np.clip(overlay_img, 0, 1)


def anim_update(i):
    fig.suptitle(f"Step: {i+1}/{T}", fontsize=16)

    # 1. 入力画像 + 入力Attention Map
    ax = axes[0]
    ax.cla()
    # k_dim個のマップを平均して重ねる
    avg_input_attn = np.mean(input_attns[i], axis=0)
    input_overlay = create_overlay(np.transpose(images[i], (1, 2, 0)), avg_input_attn)
    ax.imshow(input_overlay)
    ax.set_title("Input Img + Input Attn")
    ax.axis("off")

    # 2. 予測画像
    ax = axes[1]
    ax.cla()
    ax.imshow(pred_images[i])
    ax.set_title("Predicted Image (t+1)")
    ax.axis("off")

    # 3. 予測Attention Map
    ax = axes[2]
    ax.cla()
    avg_pred_heatmap = np.mean(pred_heatmaps[i], axis=0)
    # 予測は次のフレームの入力画像に重ねる
    pred_overlay = create_overlay(np.transpose(images[i + 1], (1, 2, 0)), avg_pred_heatmap)
    ax.imshow(pred_overlay)
    ax.set_title("Predicted Heatmap")
    ax.axis("off")

    # 4. 関節角の軌跡
    ax = axes[3]
    ax.cla()
    ax.set_title("Tip Position (xyz)")
    ax.set_xlim(0, T)
    # 3次元（xyz）に絞ってプロット
    num_joints_to_plot = 3
    # 正解軌道（破線）
    ax.plot(true_joints[1:, :num_joints_to_plot], linestyle="--", color="gray")
    # 予測軌道（実線）
    if i > 0:
        ax.plot(pred_joints[: i + 1, :num_joints_to_plot], linestyle="-")
    ax.legend(
        [f"true_{axis}" for axis in "xyz"] + [f"pred_{axis}" for axis in "xyz"], loc="upper right", fontsize="small"
    )
    ax.grid(True)


# アニメーションを生成して保存
ani = anim.FuncAnimation(fig, anim_update, frames=T, interval=100)
gif_path = os.path.join(output_folder, f"prediction_animation_idx{args.idx}.gif")
ani.save(gif_path, writer="pillow")

print(f"\nAnimation saved to: {gif_path}")
plt.close(fig)
