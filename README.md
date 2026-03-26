# 修士研究 引継ぎ資料 — ConvTAiRNN

## 概要

本研究では、ロボットアームの模倣学習のための時系列モデル **ConvTAiRNN（Convolutional Temporal Attention-integrated RNN）** を開発した。
画像・関節状態・カメラ座標を統合し、Cross-Attention + ConvLSTM により次時刻の状態を予測する。

---

## ディレクトリ構成

```
handover/
├── train_TAiRNN_conv_wo.py          # 学習スクリプト（メイン）
├── test_TAiRNN_conv_wo.py           # テスト・可視化スクリプト
├── dataset_loader_v2_wo_quat.py     # データローダー（学習用・chunk対応）
├── dataset_loader_wo_quat.py        # データローダー（テスト用）
├── convert2npz.py                   # チェックポイント → npz 変換
├── png2gif_output.py                # 出力画像列 → GIF 生成
├── libs/
│   └── convTBPTT_rep.py             # Truncated BPTT トレーナー
├── ModelCode/
│   ├── model/
│   │   ├── ConvTAiRNNv4.py          # ★ 修論メインモデル
│   │   ├── ConvTAiRNN.py ～ v8.py   # 比較・変換スクリプト用
│   │   ├── SARNN.py                 # ベースラインモデル
│   │   └── TAiRNN.py ～ v8.py       # 比較・変換スクリプト用
│   ├── layer/SpatialSoftmax.py
│   ├── data/dataset.py              # MultimodalDataset（PyTorch Dataset）
│   └── utils/                       # 各種ユーティリティ
└── ros/
    └── target_publisher_convTAiRNN_v2.py  # ロボット実機制御スクリプト（ROS）
```

`others/` に実験過程の旧バージョン全ファイルを保管。

---

## モデルアーキテクチャ：ConvTAiRNNv4

### 入力

| 変数 | 形状 | 内容 |
|------|------|------|
| `xi` | `(B, 3, H, W)` | RGB画像（H=W=128） |
| `xv` | `(B, joint_dim+3)` | 関節状態 + カメラ座標 |

`xv` の内訳（`joint_dim=4`, extra=3の場合）：

| インデックス | 内容 |
|------------|------|
| 0:3 | ΔPos 予測対象（エンドエフェクタ XYZ 差分） |
| 3 | グリッパー開閉状態 |
| 4:7 | カメラ座標系での位置（Cross-Attentionのクエリに使用） |

### 処理フロー

```
xi ──→ [token_encoder (Conv×3+BN)] ──→ KV tokens (B, N, attn_dim)
                                              │
xv_extra ──→ [state_to_ctx (MLP)] ──→ state_context
object_queries (learnable) ──────────→  Q = object_queries + state_context
                                              │
                              [Cross-Attention MHA]
                                              │
                                       enc_map (B, k_dim, Hp, Wp)
                                              │
xi ──→ [im_encoder (Conv×3)] ──→ im_hid      │
xv_joint ──────────────────────→ xv_map      │
                                        [ConvLSTMCell]
                                              │
                                       h (B, rec_dim, Hp, Wp)
                        ┌─────────────────────┤
                        │             [map_predictor (Conv)]
                        │                     │
                        │              pred_attn_map (B, k_dim, Hp, Wp)
                        │                     │
                   im_hid × pred_attn_map ──→ [decoder_image] ──→ y_image
                        │
              slot_features (h を pred_attn_map で重み付き平均プーリング)
                        │
                   [decoder_joint (MLP)] ──→ y_joint
```

### 出力

| 変数 | 内容 |
|------|------|
| `y_image` | 次時刻の再構成画像 |
| `y_joint` | 次時刻の関節状態予測（ΔPos + gripper） |
| `enc_map` | Cross-Attentionの注意マップ（B, k_dim, Hp, Wp） |
| `pred_attn_map` | ConvLSTMが予測する次時刻の注意マップ |
| `(h, c)` | ConvLSTMの隠れ状態（次ステップへ引き継ぐ） |

### 主なハイパーパラメータ

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `rec_dim` | 64 | ConvLSTM の隠れチャンネル数 |
| `k_dim` | 1 | 注意マップのチャンネル数（対象物体数に対応） |
| `joint_dim` | 4 | 関節状態次元（ΔPos 3次元 + gripper 1次元） |
| `temperature` | 5e-3 | Cross-Attentionのシャープニング温度 |
| `im_size` | 128 | 入力画像の一辺ピクセル数 |
| `attn_dim` | 64 | Cross-Attention の埋め込み次元 |

---

## 損失関数

学習には4種類の損失を加重和で使用する（`libs/convTBPTT_rep.py`）。

```
Loss = img_loss × w_img + joint_loss × w_joint + pt_loss × w_pt + rp_loss × w_rp
```

| 損失 | 内容 | 引数 |
|------|------|------|
| `img_loss` | 再構成画像の MSE | `--img_loss` |
| `joint_loss` | ΔPos の MSE + gripper BCE（重み 1e-2） | `--joint_loss` |
| `pt_loss` | `pred_attn_map[t]` vs `enc_map[t+1]` の weighted MSE | `--pt_loss` |
| `rp_loss` | k_dim個の注意マップ間のコサイン類似度ペナルティ（分散促進） | `--rp_loss` |

**ΔPos について**：`y_joint` はエンドエフェクタの**絶対位置ではなく1ステップの差分（ΔPos）**を予測する。ロボット実機制御時には累積して絶対位置に変換する。

`pt_loss` と `rp_loss` には学習初期を緩める `LossScheduler`（S字カーブ）が適用される。

---

## 学習方法：Truncated BPTT

長い時系列を `chunk_size`（デフォルト 30）ステップずつに区切り、各チャンク内で誤差逆伝播を行う。チャンク間では隠れ状態を `detach()` して渡す。

```
[t=0 ─── t=29] → backward → detach state → [t=30 ─── t=59] → backward → ...
```

---

## データ形式

### ディレクトリ構造（1エピソード = 1フォルダ）

```
dataset_root/
├── episode_001/
│   ├── image_000000.png
│   ├── image_000001.png
│   ├── ...
│   ├── position_000000.txt
│   ├── position_000001.txt
│   └── ...
└── episode_002/
    └── ...
```

### position ファイルの形式

1ファイル1行、カンマ区切りの数値列：

```
data[0], data[1], data[2]  → エンドエフェクタ XYZ（mm）
data[7]                    → グリッパー開閉（0.0～1.0）
data[9], data[10], data[11]→ カメラ座標系 XYZ
```

※ クォータニオン（data[3:7]）は本モデルでは使用しない（`_wo_quat` = without quaternion）。

### データローダーの違い

| ファイル | 用途 | 違い |
|---------|------|------|
| `dataset_loader_v2_wo_quat.py` | **学習用** | `chunk_size` 引数あり。エピソードをchunkに分割してロード |
| `dataset_loader_wo_quat.py` | **テスト用** | シーケンス全体を `max_len` でパディング。`decrement=2` 固定 |

---

## 学習手順

### 1. 学習の実行

```bash
python train_TAiRNN_conv_wo.py \
    --batch_size 4 \
    --epoch 5000 \
    --im_size 128 \
    --rec_dim 64 \
    --joint_loss 8 \
    --img_loss 1 \
    --pt_loss 0.1 \
    --rp_loss 0 \
    --lr 1e-4 \
    --k_dim 1 \
    --decrement 2 \
    --chunk_size 30 \
    --joint_dim 4 \
    --temperature 5e-3 \
    --heatmap_size 1e-2 \
    --tag 任意のタグ名
```

**注意**：`train_TAiRNN_conv_wo.py` 内の `dataset_root_dir` を自分の環境に合わせて変更すること。

### 2. 学習結果の保存先

```
log/<タグ名>/
├── SARNN_latest.pth          # 最新チェックポイント
├── SARNN_early_stop.pth      # EarlyStopping のベストモデル
├── epoch_checkpoints/
│   ├── epoch_00100.pth
│   └── ...
├── norm_params.npz            # 正規化パラメータ（data_min, data_max）
└── dataset_info.txt           # 実験条件の記録
```

TensorBoard でロスを確認：
```bash
tensorboard --logdir log/
```

### 3. テスト・可視化

```bash
python test_TAiRNN_conv_wo.py \
    --filename <タグ名> \
    --ckpt "epoch_checkpoints/epoch_01600.pth" \
    --idx 0
```

---

## ロボット実機制御（ROS）

`ros/target_publisher_convTAiRNN_v2.py` を使用。

- **`eipl` パッケージ**から `ConvTAiRNNv4` をインポートしている（`from eipl.model import ConvTAiRNNv4`）
- `ModelCode` と `eipl` は同一の実装であることを確認すること
- `data_min`, `data_max`（正規化パラメータ）をスクリプト内にハードコードしているため、**学習時の `norm_params.npz` の値と一致させること**
- カメラ画像は ROS トピックから受信し、推論結果をエンドエフェクタ目標位置として送信する

---

## 依存ライブラリ

```
torch
torchvision
numpy
Pillow
scikit-learn
matplotlib
tensorboard
tqdm
```

ROS スクリプト追加依存：
```
rospy
cv2 (opencv-python)
```

---

## 実験変遷メモ（`others/` の位置付け）

`others/` には開発過程のバージョンが保管されている。参考として主な変遷を記す。

| シリーズ | 内容 |
|---------|------|
| `train_TAiRNN_rep_*` | TAiRNN + Repulsion Loss の実験系列（v1～v16） |
| `train_ConvTAiRNN.py` | ConvLSTM + Spatial Softmax の初期版 |
| `train_SARNN_v2.py` | ベースライン（SARNN: Spatial Attention RNN） |
| `train_TAiRNN_conv_wo_v2～v6` | ConvTAiRNNv4以降の改良実験 |
| `libs/TBPTT_rep_*` | TBPTT トレーナーの各バージョン |

修論で使用した学習スクリプトは `train_TAiRNN_conv_wo.py`（`libs/convTBPTT_rep.py` + `ModelCode/model/ConvTAiRNNv4.py`）。
