import argparse
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from dataset_loader import dataset_loader

# sys.path.append("./libs/")
from libs.fullBPTT_rep_v2 import fullBPTTtrainer  # TODO
from ModelCode.data import MultimodalDataset  # TODO
from ModelCode.model import TAiRNNold  # TODO
from ModelCode.utils import EarlyStopping  # TODO
from ModelCode.utils import (
    check_args,
    load_checkpoint,
    load_model,
    seed_everything,
    set_logdir,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 変更：

parser = argparse.ArgumentParser(description="TAiRNNv2")
parser.add_argument("--model", type=str, default="sarnn")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--rec_dim", type=int, default=50)
parser.add_argument("--k_dim", type=int, default=5)
parser.add_argument("--joint_dim", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--img_loss", type=float, default=1.0)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--pt_loss", type=float, default=1.0)
parser.add_argument("--rp_loss", type=float, default=1.0)
parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)
parser.add_argument("--im_size", type=int, default=64)
parser.add_argument("--chunk_size", type=int, default=30)
parser.add_argument("--decrement", type=int, default=1)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

# check args
args = check_args(args)


# 諸々の初期設定
# 乱数初期設定
seed_everything(seed=42)
# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
# log保存先
log_dir_path = set_logdir("./" + args.log_dir, args.tag)


# データセット
dataset_root_dir = "/work/gn45/n45002/savedata/traindata_20251124_comb/"
# dataset_root_dir = "/work/gn45/n45002/savedata/traindata_20250930_v1/"
folder_names = sorted([name for name in os.listdir(dataset_root_dir) if not name.startswith(".")])

# decrement
ratio = 1
sample_size = int(len(folder_names) * ratio)
selected_folders = random.sample(folder_names, sample_size)
print(selected_folders)

train_folders, test_folders = train_test_split(selected_folders, random_state=42)
images, joints, data_min, data_max = dataset_loader(
    train_folders,
    dataset_root_dir,
    joint_dim=args.joint_dim,
    img_size=args.im_size,
)
print(f"min:{data_min}, max:{data_max}")
np.savez(os.path.join(log_dir_path, "norm_params.npz"), data_min=data_min, data_max=data_max)


min_str = ",".join(map(str, data_min))
max_str = ",".join(map(str, data_max))
output_path = os.path.join(log_dir_path, "dataset_info.txt")
with open(output_path, "w") as file:
    file.write(f"model: {args.model}\n\n")

    file.write(f"root_dir: {dataset_root_dir}\n")
    file.write(f"train_folders:{train_folders}\n")
    file.write(f"test_folders:{test_folders}\n\n")

    file.write(f"Mean: " + min_str + "\n")
    file.write(f"Std: " + max_str + "\n")
    file.write(f"Train Dataset Shape: {images.shape}\n")
    file.write(f"epoch:{args.epoch}\n\n")

    file.write(f"k_dim:{args.k_dim}\n")
    file.write(f"img_loss:{args.img_loss}\n")
    file.write(f"joint_loss:{args.joint_loss}\n")
    file.write(f"pt_loss:{args.pt_loss}\n")
    file.write(f"rp_loss:{args.rp_loss}\n")
    file.write(f"heatmap_size:{args.heatmap_size}\n")
    file.write(f"temparature_loss:{args.temperature}\n\n")

    file.write(f"joint_dim:{args.joint_dim}\n")
    file.write(f"im_size:{args.im_size}\n")
    file.write(f"chunk_size:{args.chunk_size}\n")
    file.write(f"decrement:{args.decrement}\n")
    file.write(f"loss_rate:{args.lr}\n\n")

    file.write("2025/11/19: train_rep_v8")

stdev = 0.1
train_dataset = MultimodalDataset(images, joints, device=device, stdev=stdev)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

# testデータの読み込み
images, joints, _, _ = dataset_loader(
    test_folders,
    dataset_root_dir,
    joint_dim=args.joint_dim,
    img_size=args.im_size,
    data_min=data_min,
    data_max=data_max,
)
test_dataset = MultimodalDataset(images, joints, device=device, stdev=None)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
)

del images
del joints

# TODO:argsのmodelから選択できるようにする
ModelClass = load_model(model_name="TAiRNNv2", class_name="TAiRNNv2")
model = TAiRNNold(
    rec_dim=args.rec_dim,
    joint_dim=args.joint_dim,
    k_dim=args.k_dim,
    heatmap_size=args.heatmap_size,
    kernel_size=3,
    temperature=args.temperature,
    im_size=[args.im_size, args.im_size],
)

# TODO:コード妥当性の確認
# torch.compile makes PyTorch code run faster
if args.compile:
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)


optimizer = optim.Adam(model.parameters(), eps=1e-07, lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr * 0.01)

# load trainer/tester class
loss_weights = [args.img_loss, args.joint_loss, args.pt_loss, args.rp_loss]
trainer = fullBPTTtrainer(
    model,
    optimizer,
    loss_weights=loss_weights,
    joint_dim=args.joint_dim,
    device=device,
)

### training main
save_name = os.path.join(log_dir_path, "SARNN_early_stop.pth")
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
start_epoch = load_checkpoint(save_name, model, optimizer, scheduler)
early_stop = EarlyStopping(patience=1000)

pbar_epoch = tqdm(
    range(start_epoch, args.epoch),  # ← 途中から数え直さない
    initial=start_epoch,  # ← 進捗バーも継続表示
    total=args.epoch,
)

for epoch in pbar_epoch:
    # ---------- train ----------
    (
        train_loss,
        train_img_loss,
        train_joint_loss,
        train_pt_loss,
        train_rp_loss,
        train_pos_loss,
        train_quat_loss,
        train_grip_loss,
    ) = trainer.process_epoch(train_loader)

    # ---------- test ----------
    with torch.no_grad():
        (
            test_loss,
            test_img_loss,
            test_joint_loss,
            test_pt_loss,
            test_rp_loss,
            test_pos_loss,
            test_quat_loss,
            test_grip_loss,
        ) = trainer.process_epoch(test_loader, training=False)

    # ---------- TensorBoard logging ----------
    writer.add_scalar("Loss/total_train_loss", train_loss, epoch)
    writer.add_scalar("Loss/img_train_loss", train_img_loss, epoch)
    writer.add_scalar("Loss/joint_train_loss", train_joint_loss, epoch)
    writer.add_scalar("Loss/pt_train_loss", train_pt_loss, epoch)
    writer.add_scalar("Loss/rp_train_loss", train_rp_loss, epoch)

    writer.add_scalar("Loss/pos_train_loss", train_pos_loss, epoch)
    writer.add_scalar("Loss/quat_train_loss", train_quat_loss, epoch)
    writer.add_scalar("Loss/grip_train_loss", train_grip_loss, epoch)

    writer.add_scalar("Loss/total_test_loss", test_loss, epoch)
    writer.add_scalar("Loss/img_test_loss", test_img_loss, epoch)
    writer.add_scalar("Loss/joint_test_loss", test_joint_loss, epoch)
    writer.add_scalar("Loss/pt_test_loss", test_pt_loss, epoch)
    writer.add_scalar("Loss/rp_test_loss", test_rp_loss, epoch)

    writer.add_scalar("Loss/pos_test_loss", test_pos_loss, epoch)
    writer.add_scalar("Loss/quat_test_loss", test_quat_loss, epoch)
    writer.add_scalar("Loss/grip_test_loss", test_grip_loss, epoch)

    writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

    # ---------- save latest ----------
    latest_save_name = os.path.join(log_dir_path, "SARNN_latest.pth")
    trainer.save(epoch, [train_loss, test_loss], latest_save_name, scheduler.state_dict())

    # ---------- early stop ----------
    save_ckpt, _ = early_stop(test_loss)
    if save_ckpt:
        trainer.save(epoch, [train_loss, test_loss], save_name, scheduler.state_dict())
    scheduler.step()

    # ---------- progress bar ----------
    pbar_epoch.set_postfix(
        OrderedDict(
            train_loss=train_loss,
            test_loss=test_loss,
            train_img_loss=train_img_loss,
            test_img_loss=test_img_loss,
            train_joint_loss=train_joint_loss,
            test_joint_loss=test_joint_loss,
            train_pt_loss=train_pt_loss,
            test_pt_loss=test_pt_loss,
            train_rp_loss=train_rp_loss,
            test_rp_loss=test_rp_loss,
            train_pos_loss=train_pos_loss,
            test_pos_loss=test_pos_loss,
            train_quat_loss=train_quat_loss,
            test_quat_loss=test_quat_loss,
            train_grip_loss=train_grip_loss,
            test_grip_loss=test_grip_loss,
            lr=optimizer.param_groups[0]["lr"],
        )
    )
output_path = os.path.join(log_dir_path, "UAA_UAG_UGA.txt")
with open(output_path, "w") as file:
    file.write(f"finish\n")
