import argparse
import os

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
root_dir = "/work/gn45/n45002/savedata/testdata_20250621_v2/"
# root_dir = '/work/gn45/n45002/savedata/savedata_test/'
testdata_folder = sorted([name for name in os.listdir(root_dir)])
testdata_folder = [testdata_folder[15]]
print(testdata_folder)


# dataset_loaderの呼び出しを修正
images, joints, _, _ = dataset_loader(testdata_folder, root_dir, data_min=data_min, data_max=data_max)
images = images[0]
joints = joints[0]

# 非正規化（逆変換）の式を修正
true_joints = joints * (data_max - data_min) + data_min

img_size = params["im_size"]

output_folder = str(testdata_folder) + args.filename


# define model
model = TAiRNN(
    rec_dim=params["rec_dim"],
    joint_dim=params["joint_dim"],
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
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
    pred_joint = pred_joint * (data_max - data_min) + data_min
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


def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(np.transpose(images[i, :, :], (1, 2, 0)))
    for j in range(params["k_dim"]):
        ax[0].plot(ect_pts[i, j, 0], ect_pts[i, j, 1], "bo", markersize=6)  # encoder
        ax[0].plot(dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2)  # decoder
    ax[0].axis("off")
    ax[0].set_title("Input image")

    # plot predicted image
    #     pred_image[i, :, :, :] = np.clip(pred_image[i, :, :, :], 0, 1) * 255
    ax[1].imshow(pred_image[i, :, :, :])
    ax[1].axis("off")
    ax[1].set_title("Predicted image")

    # plot joint angle
    #     ax[2].set_ylim(-1.0, 2.0)
    ax[2].set_xlim(0, T)
    ax[2].plot(true_joints[1:], linestyle="dashed", c="k")
    for joint_idx in range(3):
        ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[2].set_xlabel("Step")
    ax[2].set_title("Tip position")


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
# ani.save("./output/SARNN_{}_{}_{}.gif".format(params["tag"], idx, args.input_param))
ani.save(
    "./output/SARNN_{}_{}_{}_{}.gif".format(params["tag"], idx, args.input_param, testdata_folder[0]), writer="pillow"
)


fig1, ax1 = plt.subplots(1, 2, figsize=(8, 4))


def anim_update_1(i):
    # まずサブプロットをクリア
    ax1[0].cla()
    ax1[1].cla()

    ax1[0].imshow(np.transpose(images[i, :, :], (1, 2, 0)))
    for j in range(params["k_dim"]):
        ax1[0].plot(ect_pts[i, j, 0], ect_pts[i, j, 1], "bo", markersize=6)  # encoder
        ax1[0].plot(dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2)  # decoder
    ax1[0].axis("off")
    ax1[0].set_title("Input image")

    # plot predicted image
    #     pred_image[i, :, :, :] = np.clip(pred_image[i, :, :, :], 0, 1) * 255
    ax1[1].imshow(pred_image[i, :, :, :])
    ax1[1].axis("off")
    ax1[1].set_title("Predicted image")


ani1 = anim.FuncAnimation(fig1, anim_update_1, interval=int(np.ceil(T / 10)), frames=T)

fig2, ax2 = plt.subplots(1, 1, figsize=(12, 4))


def anim_update_2(i):
    ax2[0].cla()
    ax2[1].cla()
    # ax2[2].cla()

    ax2[0].set_xlim(0, T)
    ax2[1].set_xlim(0, T)
    # ax2[2].set_xlim(0, T)

    ax2[0].plot(true_joints[1:, :3], linestyle="dashed", c="k")
    ax2[1].plot(true_joints[1:, 3:7], linestyle="dashed", c="k")
    # ax2[2].plot(true_joints[1:, 7], linestyle="dashed", c="k")

    for joint_idx in range(3):
        ax2[0].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    for joint_idx in range(3, 7):
        ax2[1].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    # ax2[2].plot(np.arange(i + 1), pred_joint[: i + 1, 7])
    ax2[0].set_xlabel("Step")
    ax2[1].set_xlabel("Step")
    # ax2[2].set_xlabel("Step")
    ax2[0].set_title("Tip position")
    ax2[1].set_title("Quaternion")
    # ax2[2].set_title("Gripper ratio")


ani2 = anim.FuncAnimation(fig2, anim_update_2, interval=int(np.ceil(T / 10)), frames=T)


# ===============================
# 3) それぞれを保存
# ===============================
ani1.save(
    "./output/SARNN_{}_{}_{}_{}_ani1.gif".format(params["tag"], idx, args.input_param, testdata_folder[0]),
    writer="pillow",
)
ani2.save(
    "./output/SARNN_{}_{}_{}_{}_ani2.gif".format(params["tag"], idx, args.input_param, testdata_folder[0]),
    writer="pillow",
)

plt.show()


# If an error occurs in generating the gif animation or mp4, change the writer (imagemagick/ffmpeg).
# ani.save("./output/PCA_SARNN_{}.gif".format(params["tag"]), writer="imagemagick")
# ani.save("./output/PCA_SARNN_{}.mp4".format(params["tag"]), writer="ffmpeg")
