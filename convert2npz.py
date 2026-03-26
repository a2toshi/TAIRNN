import os
from collections import OrderedDict

import numpy as np
import torch
from ModelCode.model import (
    SARNN,
    ConvTAiRNN,
    ConvTAiRNNv2,
    ConvTAiRNNv3,
    ConvTAiRNNv4,
    ConvTAiRNNv8,
    TAiRNN,
    TAiRNNv2,
    TAiRNNv3,
    TAiRNNv5,
    TAiRNNv8,
)
from ModelCode.utils import restore_args

base_dir = "/work/gn45/n45002/TAiRNN/log/20260104_1300_00/"
# filename = base_dir + "SARNN_early_stop.pth"
filename = base_dir + "SARNN_latest.pth"
filenamev2 = filename
filenamev2 = base_dir + "epoch_checkpoints/epoch_02000.pth"

dir_name = os.path.split(filename)[0]

# 3. モデルのパラメータ(args.json)を復元
params = restore_args(os.path.join(dir_name, "args.json"))

# 1. モデルのインスタンスを生成
# model = ConvTAiRNN(
#     rec_dim=params["rec_dim"],
#     joint_dim=params["joint_dim"],  # 元コードの joint_dim
#     k_dim=params["k_dim"],
#     temperature=params["temperature"],
#     im_size=[64, 64],  # 元コードの img_size
# )
model = ConvTAiRNNv8(
    rec_dim=params["rec_dim"],
    joint_dim=params["joint_dim"],  # 元コードの joint_dim
    k_dim=params["k_dim"],
    temperature=params["temperature"],
    im_size=[params["im_size"], params["im_size"]],  # 元コードの img_size
)


ckpt = torch.load(filenamev2, map_location=torch.device("cpu"), weights_only=False)
state_dict = ckpt["model_state_dict"]

# --- ここからが変換処理 ---

# 6. state_dict内の各テンサーをNumPy配列に変換
numpy_state_dict = OrderedDict()
for key, tensor in state_dict.items():
    numpy_state_dict[key] = tensor.cpu().numpy()
    print(f"Processing layer: {key}")

# 7. NumPyの.npz形式で保存
np.savez(base_dir + "model_weights_2000.npz", **numpy_state_dict)

print("\n✅ Successfully converted and saved weights to model_weights.npz")
