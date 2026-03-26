import argparse
import os

import matplotlib.animation as anim
import matplotlib.pylab as plt
import numpy as np
import torch
from dataset_loader import dataset_loader
from ModelCode.model import TAiRNNv3
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

# model_folder = "./log/" + args.filename + "/SARNN_early_stop.pth"
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


# load dataset
root_dir = "/work/gn45/n45002/savedata/testdata_20250930_v1/"
# root_dir = '/work/gn45/n45002/savedata/savedata_test/'
testdata_folder = sorted([name for name in os.listdir(root_dir)])
testdata_folder = [testdata_folder[idx]]
print(testdata_folder)

joint_dim = params["joint_dim"]
img_size = params["im_size"]

# dataset_loaderの呼び出しを修正
images, joints, _, _ = dataset_loader(
    testdata_folder, root_dir, data_min=data_min, data_max=data_max, joint_dim=params["joint_dim"], img_size=img_size
)
images = images[0]
joints = joints[0]

# 非正規化（逆変換）の式を修正
true_joints = joints * (data_max - data_min) + data_min


output_folder = str(testdata_folder) + args.filename


# define model
model = TAiRNNv3(
    rec_dim=params["rec_dim"],
    joint_dim=params["joint_dim"],
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    kernel_size=3,
    temperature=params["temperature"],
    im_size=[img_size, img_size],
    visualize_on_forward=True,
    folder_name=str(testdata_folder),
    log_name=args.filename,
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
    # ここまではそのまま
    y_image, y_joint, ect_pts, dec_pts, state = model(img_t, joint_t, state)

    # --- image 復元 ---
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image, 0, 1)
    pred_image = pred_image.transpose(1, 2, 0)

    # --- joint 復元 ---
    device = joint_t.device
    joint_dim = params["joint_dim"]  # 8 のはず

    # joint_t: (B, 11) = [pos(3), quat(4), grip(1), cam(3)]
    joint_t_joint = joint_t[:, :joint_dim]  # (B, 8) だけ使う

    # data_min/max は numpy 想定なので torch に変換
    data_min_t = torch.from_numpy(data_min[:joint_dim]).to(device).float()  # (8,)
    data_max_t = torch.from_numpy(data_max[:joint_dim]).to(device).float()  # (8,)
    joint_range = data_max_t - data_min_t  # (8,)

    # 現在の物理スケール joint
    joint_curr_phys = joint_t_joint * joint_range + data_min_t  # (B, 8)
    pos_curr = joint_curr_phys[:, :3]  # (B, 3)

    # モデル出力
    delta_pos_phys = y_joint[:, :3]  # (B, 3) 物理Δpos
    other_norm = y_joint[:, 3:joint_dim]  # (B, 5) 正規化 quat+grip

    # 次ステップ位置
    pos_next = pos_curr + delta_pos_phys  # (B, 3)

    # quat+grip を逆正規化
    data_min_other_t = torch.from_numpy(data_min[3:joint_dim]).to(device).float()  # (5,)
    data_max_other_t = torch.from_numpy(data_max[3:joint_dim]).to(device).float()  # (5,)
    other_range = data_max_other_t - data_min_other_t  # (5,)

    other_phys = other_norm * other_range + data_min_other_t  # (B, 5)

    # 結合して numpy へ
    pred_joint_tensor = torch.cat([pos_next, other_phys], dim=1)  # (B, 8)
    pred_joint = tensor2numpy(pred_joint_tensor[0].detach())

    # append data
    image_list.append(pred_image)
    joint_list.append(pred_joint)
    ect_pts_list.append(tensor2numpy(ect_pts[0]))
    dec_pts_list.append(tensor2numpy(dec_pts[0]))

    print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

pred_image = np.array(image_list)
pred_joint = np.array(joint_list)

# 追加
eval_T = min(pred_joint.shape[0], true_joints.shape[0] - 1)
gt_eval = true_joints[1 : 1 + eval_T, :joint_dim]  # (eval_T, joint_dim)  t+1 のGT
pred_eval = pred_joint[:eval_T, :joint_dim]  # (eval_T, joint_dim)  モデル予測

# -------------------------
# モデル予測の誤差統計
# -------------------------
error = pred_eval - gt_eval  # (eval_T, joint_dim)

bias = np.mean(error, axis=0)  # 平均誤差
std = np.std(error, axis=0)  # 標準偏差
mae = np.mean(np.abs(error), axis=0)  # MAE
rmse = np.sqrt(np.mean(error**2, axis=0))  # RMSE

print("===== Joint prediction error statistics (MODEL, per dimension) =====")
for d in range(joint_dim):
    print(f"[dim {d}] " f"bias={bias[d]:.6f}, " f"std={std[d]:.6f}, " f"MAE={mae[d]:.6f}, " f"RMSE={rmse[d]:.6f}")

# -------------------------
# Δpos = 0 ベースライン
#   ->  q_{t+1}^base = q_t
# -------------------------
baseline_pred = true_joints[:eval_T, :joint_dim]  # 1ステップ前のGTをそのまま
baseline_error = baseline_pred - gt_eval  # (eval_T, joint_dim)

base_bias = np.mean(baseline_error, axis=0)
base_std = np.std(baseline_error, axis=0)
base_mae = np.mean(np.abs(baseline_error), axis=0)
base_rmse = np.sqrt(np.mean(baseline_error**2, axis=0))

print("===== Baseline (Δpos = 0) error statistics (per dimension) =====")
for d in range(joint_dim):
    print(
        f"[dim {d}] "
        f"bias={base_bias[d]:.6f}, "
        f"std={base_std[d]:.6f}, "
        f"MAE={base_mae[d]:.6f}, "
        f"RMSE={base_rmse[d]:.6f}"
    )

# （任意）改善度も見たいなら：
print("===== Improvement over baseline (RMSE_base - RMSE_model) =====")
for d in range(joint_dim):
    print(f"[dim {d}] ΔRMSE = {base_rmse[d] - rmse[d]:.6f}")

# ここから下はそのまま
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

folder_name = "./output/" + args.filename + "/" + str(testdata_folder) + "/joint_error/"
os.makedirs(folder_name, exist_ok=True)

steps = np.arange(eval_T)

for d in range(joint_dim):
    fig_d, axs_d = plt.subplots(2, 1, figsize=(8, 6), dpi=80)

    # 上段：GT vs Pred
    axs_d[0].plot(steps, gt_eval[:, d], label="GT")
    axs_d[0].plot(steps, pred_eval[:, d], label="Pred")
    axs_d[0].plot(steps, baseline_pred[:, d], label="Baseline")
    axs_d[0].set_title(f"Joint dim {d} - GT vs Pred")
    axs_d[0].set_xlabel("Step")
    axs_d[0].set_ylabel("Value")
    axs_d[0].legend()
    axs_d[0].grid(True)

    # 下段：誤差推移
    axs_d[1].plot(steps, error[:, d])
    axs_d[1].set_title(f"Joint dim {d} - Error (Pred - GT)")
    axs_d[1].set_xlabel("Step")
    axs_d[1].set_ylabel("Error")
    axs_d[1].grid(True)

    fig_d.tight_layout()
    save_path = folder_name + f"joint_dim{d}_error.png"
    fig_d.savefig(save_path, bbox_inches="tight")
    plt.close(fig_d)

    print(f"Saved joint error plot for dim {d} -> {save_path}")


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
    folder_name + "SARNN_{}_{}_{}_{}.gif".format(params["tag"], idx, args.input_param, testdata_folder[0]),
    writer="pillow",
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
    folder_name + "SARNN_{}_{}_{}_{}_ani1.gif".format(params["tag"], idx, args.input_param, testdata_folder[0]),
    writer="pillow",
)
ani2.save(
    folder_name + "SARNN_{}_{}_{}_{}_ani2.gif".format(params["tag"], idx, args.input_param, testdata_folder[0]),
    writer="pillow",
)

plt.show()
