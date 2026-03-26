import argparse
import os
from pathlib import Path

import matplotlib.animation as anim
import matplotlib.pylab as plt
import numpy as np
import torch
from dataset_loader import dataset_loader
from ModelCode.model import SARNN, TAiRNN
from ModelCode.utils import deprocess_img, restore_args, tensor2numpy

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
parser.add_argument("--idx", type=str, default="0")
parser.add_argument("--input_param", type=float, default=1.0)
parser.add_argument("--pretrained", action="store_true")
args = parser.parse_args()

# check args
assert args.filename or args.pretrained, "Please set filename or pretrained"

model_folder = "./log/" + args.filename + "/SARNN_latest.pth"

# restore parameters
dir_name = os.path.split(model_folder)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
idx = int(args.idx)

# train.pyで保存した正規化パラメータを自動で読み込む
norm_params_path = os.path.join(dir_name, "norm_params.npz")
if not os.path.exists(norm_params_path):
    raise FileNotFoundError(f"norm_params.npz not found in {dir_name}. Please run training first.")
norm_data = np.load(norm_params_path)
data_min = norm_data["data_min"]
data_max = norm_data["data_max"]
print("Loaded normalization parameters (min/max) successfully.")


# data_min = [
#     6.96396484e02,
#     -8.67640080e01,
#     -4.73438141e02,
#     6.78398000e-01,
#     1.00643000e-01,
#     -6.80140000e-01,
#     9.77420000e-02,
#     1.30819033e03,
#     -1.06682759e02,
#     -5.09518300e01,
# ]
# data_max = [
#     7.65542603e02,
#     -1.98433130e01,
#     -4.32622620e02,
#     8.43861000e-01,
#     3.27885000e-01,
#     -4.22925000e-01,
#     3.38767000e-01,
#     1.48631634e03,
#     4.56958690e01,
#     2.20849010e02,
# ]
# data_min = np.array(data_min)
# data_max = np.array(data_max)


# load dataset
root_dir = "/work/gn45/n45002/savedata/testdata_20250930_v1/"
# root_dir = '/work/gn45/n45002/savedata/savedata_test/'
testdata_folder = sorted([name for name in os.listdir(root_dir)])
testdata_folder = [testdata_folder[idx]]
print(testdata_folder)

joint_dim = params["joint_dim"]

# dataset_loaderの呼び出しを修正
images, joints, _, _ = dataset_loader(
    testdata_folder, root_dir, data_min=data_min, data_max=data_max, joint_dim=params["joint_dim"]
)
images = images[0]
joints = joints[0]

# 非正規化（逆変換）の式を修正
true_joints = joints * (data_max - data_min) + data_min

img_size = params["im_size"]

output_folder = str(testdata_folder[0]) + args.filename


# define model
model = TAiRNN(
    rec_dim=params["rec_dim"],
    joint_dim=params["joint_dim"],
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    kernel_size=3,
    temperature=params["temperature"],
    im_size=[img_size, img_size],
    visualize_on_forward=True,
    test_name=output_folder,
)

# load weight
ckpt = torch.load(model_folder, map_location=torch.device("cpu"), weights_only=False)
# print(ckpt.keys())
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# Inference
image_list, joint_list = [], []
ect_pts_list, dec_pts_list = [], []
state = None
nloop = len(images)
for loop_ct in range(nloop):
    # load data and normalization
    img_t = images[loop_ct]
    img_t = torch.Tensor(np.expand_dims(img_t, 0))
    joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))

    # predict rnn
    y_image, y_joint, ect_pts, dec_pts, state = model(img_t, joint_t, state)

    # denormalization
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image, 0, 1)
    pred_image = pred_image.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = pred_joint * (data_max[:joint_dim] - data_min[:joint_dim]) + data_min[:joint_dim]

    # append data
    image_list.append(pred_image)
    joint_list.append(pred_joint)
    ect_pts_list.append(tensor2numpy(ect_pts[0]))
    dec_pts_list.append(tensor2numpy(dec_pts[0]))

    print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

pred_image = np.array(image_list)
pred_joint = np.array(joint_list)

# split key points
ect_pts = np.array(ect_pts_list)
dec_pts = np.array(dec_pts_list)
ect_pts = ect_pts.reshape(-1, params["k_dim"], 2) * img_size
dec_pts = dec_pts.reshape(-1, params["k_dim"], 2) * img_size
enc_pts = np.clip(ect_pts, 0, img_size)
dec_pts = np.clip(dec_pts, 0, img_size)


plt.imshow(np.transpose(images[0, :, :], (1, 2, 0)))
# plot images
T = len(images)
fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=60)


true_pos, true_quat, true_grip = true_joints[:, :3], true_joints[:, 3:7], true_joints[:, 7]
pred_pos, pred_quat, pred_grip = pred_joint[:, :3], pred_joint[:, 3:7], pred_joint[:, 7]


def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # (1) Input image + keypoints
    ax[0].imshow(np.transpose(images[i, :, :], (1, 2, 0)))
    for j in range(params["k_dim"]):
        ax[0].plot(ect_pts[i, j, 0], ect_pts[i, j, 1], "o", markersize=6)
        ax[0].plot(dec_pts[i, j, 0], dec_pts[i, j, 1], "x", markersize=6)
    ax[0].axis("off")
    ax[0].set_title("Input image")

    # (2) Predicted image
    ax[1].imshow(pred_image[i])
    ax[1].axis("off")
    ax[1].set_title("Predicted image")

    # (3) Position (x,y,z)
    ax[2].set_xlim(0, T - 1)
    ax[2].plot(np.arange(T), true_pos[:, 0], linestyle="dashed")
    ax[2].plot(np.arange(T), true_pos[:, 1], linestyle="dashed")
    ax[2].plot(np.arange(T), true_pos[:, 2], linestyle="dashed")
    ax[2].plot(np.arange(i + 1), pred_pos[: i + 1, 0])
    ax[2].plot(np.arange(i + 1), pred_pos[: i + 1, 1])
    ax[2].plot(np.arange(i + 1), pred_pos[: i + 1, 2])
    ax[2].set_xlabel("Step")
    ax[2].set_title("Tip position (x,y,z)")


save_dir = Path("./output") / params["tag"] / testdata_folder[0]
ani = anim.FuncAnimation(fig, anim_update, interval=max(10, int(np.ceil(T / 10))), frames=T)
os.makedirs("./output", exist_ok=True)
filename = f"SARNN_{idx}_{args.input_param}_overview.gif"
ani.save(save_dir / filename, writer="pillow")

# ---------- 個別アニメ：Position / Quaternion / Gripper ----------
fig_state, ax_state = plt.subplots(1, 3, figsize=(14, 4), dpi=80)


def anim_update_state(i):
    for a in ax_state:
        a.cla()

    # Pos
    ax_state[0].set_xlim(0, T - 1)
    for d in range(3):
        ax_state[0].plot(np.arange(T), true_pos[:, d], linestyle="dashed")
        ax_state[0].plot(np.arange(i + 1), pred_pos[: i + 1, d])
    ax_state[0].set_title("Position (x,y,z)")
    ax_state[0].set_xlabel("Step")

    # Quaternion
    ax_state[1].set_xlim(0, T - 1)
    for d, lbl in enumerate(["qx", "qy", "qz", "qw"]):
        ax_state[1].plot(np.arange(T), true_quat[:, d], linestyle="dashed")
        ax_state[1].plot(np.arange(i + 1), pred_quat[: i + 1, d])
    ax_state[1].set_title("Quaternion (qx,qy,qz,qw)")
    ax_state[1].set_xlabel("Step")

    # Gripper
    ax_state[2].set_xlim(0, T - 1)
    ax_state[2].plot(np.arange(T), true_grip, linestyle="dashed", label="true")
    ax_state[2].plot(np.arange(i + 1), pred_grip[: i + 1], label="pred")
    ax_state[2].set_title("Gripper opening")
    ax_state[2].set_xlabel("Step")


ani_state = anim.FuncAnimation(fig_state, anim_update_state, interval=max(10, int(np.ceil(T / 10))), frames=T)
filename = f"SARNN_{idx}_{args.input_param}_state.gif"
ani_state.save(save_dir / filename, writer="pillow")


# ---------- 静止画（PNG）も保存：最終時点までの全曲線 ----------
def save_png_curves():
    # Position
    figp, axp = plt.subplots(figsize=(6, 4), dpi=120)
    for d, lbl in enumerate(["x", "y", "z"]):
        axp.plot(true_pos[:, d], linestyle="dashed")
        axp.plot(pred_pos[:, d])
    axp.set_title("Position (x,y,z)")
    axp.set_xlabel("Step")
    figp.tight_layout()
    filename = f"SARNN_{idx}_{args.input_param}_pos.png"
    figp.savefig(save_dir / filename)

    # Quaternion
    figq, axq = plt.subplots(figsize=(6, 4), dpi=120)
    for d, lbl in enumerate(["qx", "qy", "qz", "qw"]):
        axq.plot(true_quat[:, d], linestyle="dashed")
        axq.plot(pred_quat[:, d])
    axq.set_title("Quaternion (qx,qy,qz,qw)")
    axq.set_xlabel("Step")
    figq.tight_layout()
    filename = f"SARNN_{idx}_{args.input_param}_quat.png"
    figq.savefig(save_dir / filename)

    # Gripper
    figg, axg = plt.subplots(figsize=(6, 4), dpi=120)
    axg.plot(true_grip, linestyle="dashed", label="true")
    axg.plot(pred_grip, label="pred")
    axg.set_title("Gripper opening")
    axg.set_xlabel("Step")
    figg.tight_layout()
    filename = f"SARNN_{idx}_{args.input_param}_grip.png"
    figg.savefig(save_dir / filename)


save_png_curves()

plt.show()
