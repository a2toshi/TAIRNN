# 実験実施ガイド — ConvTAiRNN

このドキュメントでは、実際にコードを動かして実験を進める手順を説明する。
「データを用意してから学習・評価・ロボット実行まで」を一通りカバーする。

---

## 目次

1. [環境セットアップ](#1-環境セットアップ)
2. [データの収集と保存形式](#2-データの収集と保存形式)
3. [データの配置場所](#3-データの配置場所)
4. [学習の実行](#4-学習の実行)
5. [学習結果の出力と確認方法](#5-学習結果の出力と確認方法)
6. [テスト・可視化の実行](#6-テスト可視化の実行)
7. [ハイパーパラメータの調整指針](#7-ハイパーパラメータの調整指針)
8. [チェックポイントの変換（npz）](#8-チェックポイントの変換npz)
9. [ロボット実機への適用](#9-ロボット実機への適用)

---

## 1. 環境セットアップ

```bash
pip install torch torchvision numpy Pillow scikit-learn matplotlib tensorboard tqdm
```

学習・テストスクリプトは `handover/` をカレントディレクトリとして実行すること。

```bash
cd path/to/handover/
python train_TAiRNN_conv_wo.py --tag テスト実験
```

---

## 2. データの収集と保存形式

### フォルダ構造（1デモ = 1フォルダ）

```
dataset_root/
├── episode_001/
│   ├── image_000000.png
│   ├── image_000001.png
│   ├── image_000002.png
│   │   ...
│   ├── position_000000.txt
│   ├── position_000001.txt
│   ├── position_000002.txt
│   │   ...
├── episode_002/
│   └── ...
└── episode_NNN/
```

- `image_XXXXXX.png`：RGBカメラ画像。**ファイル名は `image` で始まること**（ローダーが `startswith("image")` でフィルタリングするため）
- `position_XXXXXX.txt`：ロボット状態。**ファイル名は `position` で始まること**
- ファイルは `sorted()` で昇順に読まれるため、ゼロ埋めして連番にすること

### position ファイルの形式

1ファイルにつき1行、カンマ区切り：

```
x,y,z,qw,qx,qy,qz,gripper,_,cam_x,cam_y,cam_z,...
 0  1  2   3   4   5   6     7  8    9     10    11
```

データローダーが実際に使う列：

| インデックス | 内容 | 単位 |
|------------|------|------|
| 0, 1, 2 | エンドエフェクタ XYZ | mm |
| 7 | グリッパー開閉比 | 0.0（閉）〜1.0（開） |
| 9, 10, 11 | カメラ座標系 XYZ | mm |

クォータニオン（3〜6列）は **使用しない**（`_wo_quat` = without quaternion）。

### 画像のクロップ・リサイズ

データローダー内でクロップ・リサイズが自動で行われる：

```python
crop_box = (160, 0, 1440, 1000)   # 元画像からこの領域を切り出す
img_size = 128                     # 切り出した後にこのサイズへリサイズ
```

**元画像の解像度が異なる場合は `dataset_loader_v2_wo_quat.py` の `crop_box` を変更すること。**

### データ量の目安

修論での実績：
- エピソード数：20〜30エピソード（train/test = 3:1 分割）
- 1エピソードあたりのステップ数：〜150ステップ程度
- `decrement=2`（1ステップおきに間引き）を使用

---

## 3. データの配置場所

`train_TAiRNN_conv_wo.py` 内の以下の行を自分の環境に合わせて書き換える：

```python
# ---- train_TAiRNN_conv_wo.py 内 ----
dataset_root_dir = "/work/gn45/n45002/savedata/traindata_20251213_v1/"   # ← ここを変更
```

テストスクリプトも同様：

```python
# ---- test_TAiRNN_conv_wo.py 内 ----
root_dir = "/work/gn45/n45002/savedata/testdata_20251213_v1/"   # ← ここを変更
```

**学習データとテストデータは別フォルダに分けて管理することを推奨。**
スクリプト内で `train_test_split` を使って学習/テスト分割もしているが、
ここでは学習時に使わなかったエピソードを別フォルダに置いて評価した。

---

## 4. 学習の実行

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
    --tag 20251220_1500_00
```

### `--tag` について

- ログの保存先フォルダ名になる：`log/<tag>/`
- 日付時刻形式（`YYYYMMDD_HHMM_SS`）を推奨。`--tag` を省略すると自動で日時が使われる
- **同じタグで再実行すると `SARNN_early_stop.pth` を読み込んで学習を再開する**（途中再開機能あり）

### GPU の使用

CUDA が使える環境では自動的に GPU が使われる。使用デバイスは標準出力に表示される。

---

## 5. 学習結果の出力と確認方法

### 保存先ディレクトリ

```
log/<tag>/
├── args.json                       # 実行時の全引数（テスト時に自動で読み込まれる）
├── norm_params.npz                 # 正規化パラメータ（data_min, data_max）
├── dataset_info.txt                # データセット構成のメモ
├── SARNN_latest.pth                # 毎エポック上書き保存される最新チェックポイント
├── SARNN_early_stop.pth            # テストlossが改善したときのみ保存されるベストモデル
└── epoch_checkpoints/
    ├── epoch_00100.pth
    ├── epoch_00200.pth
    └── ...                         # 100エポックごとに保存
```

### TensorBoard でのロス確認

```bash
tensorboard --logdir log/
```

ブラウザで `http://localhost:6006` を開く。

確認すべき主なグラフ：

| グラフ名 | 確認ポイント |
|---------|-------------|
| `Loss/total_train_loss` | 全体のlossが下がっているか |
| `Loss/total_test_loss` | train lossと乖離していないか（過学習チェック） |
| `Loss/img_train_loss` | 画像再構成ができているか |
| `Loss/joint_train_loss` | 関節状態予測ができているか |
| `Loss/pos_train_loss` | ΔPos予測（X/Y/Z）の収束 |
| `Loss/grip_train_loss` | グリッパー予測の収束 |
| `LearningRate` | CosineAnnealingが正常に動いているか |

### 学習が上手くいっているかの判断基準

1. **`total_test_loss` が単調減少しているか**（増加に転じたら過学習）
2. **`img_loss` が `joint_loss` よりも速く下がる** ことが多い
3. **`pos_train_loss` が xyz 方向で均等に下がっているか**（特定軸だけ大きいなら要調整）
4. train と test の差が大きい → バッチサイズを下げるかデータを増やす

---

## 6. テスト・可視化の実行

```bash
python test_TAiRNN_conv_wo.py \
    --filename 20251220_1500_00 \
    --ckpt "epoch_checkpoints/epoch_01600.pth" \
    --idx 0
```

| 引数 | 説明 |
|------|------|
| `--filename` | `log/` 以下のタグ名（学習時の `--tag` と同じ） |
| `--ckpt` | 使用するチェックポイント（`log/<tag>/` からの相対パス） |
| `--idx` | テストデータのエピソードインデックス（0始まり） |

テストデータのパスは **スクリプト内の `root_dir` を直接書き換える**。

### テスト結果の出力先

```
output/<tag>/<ckpt名>/<テストフォルダ名>/
├── joint_error/
│   ├── joint_error.txt             # RMSE・MAE・バイアスの統計テキスト
│   ├── joint_dim0_error.png        # 各次元のGT vs 予測グラフ
│   ├── joint_dim1_error.png
│   ├── joint_dim2_error.png
│   ├── joint_dim3_error.png
│   └── SARNN_<tag>_0_1.0_<episode>.gif   # 推論アニメーション（画像+関節）
└── step_XXXXXX_b0/                 # Attention Map の可視化（フォワード時に生成）
    ├── input.png
    ├── attn_w_grid.png
    ├── attn_w_grid_overlay.png
    ├── enc_map_grid.png
    └── pred_attn_map_grid.png
```

### 評価指標の読み方（`joint_error.txt`）

```
===== Joint prediction error statistics (MODEL, per dimension) =====
[dim 0] bias=..., std=..., MAE=..., RMSE=...   # X方向 ΔPos
[dim 1] bias=..., std=..., MAE=..., RMSE=...   # Y方向 ΔPos
[dim 2] bias=..., std=..., MAE=..., RMSE=...   # Z方向 ΔPos
[dim 3] bias=..., std=..., MAE=..., RMSE=...   # グリッパー

===== Baseline (Δpos = 0) error statistics (per dimension) =====
...（ΔPos=0 を予測し続けた場合のベースライン）

===== Improvement over baseline (RMSE_base - RMSE_model) =====
...（正の値 → モデルがベースラインより良い）
```

**ベースライン（Δpos=0）に対して RMSE が改善していれば、モデルが有効に動作している。**

---

## 7. ハイパーパラメータの調整指針

### 損失重みのバランス

修論での最終設定：`--img_loss 1 --joint_loss 8 --pt_loss 0.1 --rp_loss 0`

| 引数 | 増やすと | 使用上の注意 |
|------|---------|-------------|
| `--img_loss` | 再構成画像の精度↑、関節予測が疎かになる | 1〜5 の範囲が多い |
| `--joint_loss` | 関節予測の精度↑、画像学習が不安定になりうる | 8 が最終値 |
| `--pt_loss` | Attention Mapの時系列一貫性↑ | 大きすぎると不安定 |
| `--rp_loss` | k_dim個のAttentionが分散↑ | **k_dim=1 の場合は 0 にする**（意味がない） |

### k_dim（注意マップ数）

- **追跡したい物体の数**に対応させる
- k_dim=1：ターゲット1つを追跡（修論最終設定）
- k_dim=2〜4：複数物体がある場合。増やすと `rp_loss` が有効になる

### temperature

- Cross-Attention の Softmax のシャープさを制御
- 小さい（例: 1e-4）→ 注意が1点に集中、大きい（例: 1e0）→ 拡散
- 修論最終設定: `5e-3`

### rec_dim

- ConvLSTM の隠れチャンネル数。大きいほど表現力↑、計算コスト↑
- 修論設定: 64

### chunk_size と decrement

| 引数 | 説明 |
|------|------|
| `--chunk_size` | Truncated BPTT の1チャンク長。長いほど長期依存を学習できるが VRAM 消費↑ |
| `--decrement` | データの間引き率。2 なら1ステップおき（実効フレームレートが半分になる） |

### 学習率スケジューリング

CosineAnnealing を使用（`T_max=epoch`, `eta_min=lr*0.01`）。
学習が途中で発散する場合は `--lr` を 1e-5 程度に下げる。

---

## 8. チェックポイントの変換（npz）

`convert2npz.py` を使うと、`.pth` ファイルの重みを `.npz`（NumPy形式）に変換できる。
eipl などの別環境でモデル重みを使いたい場合に使用。

スクリプト内の以下を変更して実行：

```python
# convert2npz.py 内
base_dir = "/path/to/log/<tag>/"
filename = base_dir + "SARNN_latest.pth"          # 変換したい .pth ファイル
```

```bash
python convert2npz.py
```

出力：`log/<tag>/model_weights_<epoch>.npz`

---

## 9. ロボット実機への適用

`ros/target_publisher_convTAiRNN_v2.py` を使用（ROS1 環境を想定）。

### 実行前に確認すること

#### (1) モデルクラスの整合性

スクリプトは `from eipl.model import ConvTAiRNNv4` でインポートしている。
`eipl` の `ConvTAiRNNv4` と `ModelCode/model/ConvTAiRNNv4.py` が同一実装であることを必ず確認すること。

#### (2) 正規化パラメータの一致

スクリプト内の `data_min`, `data_max` が、学習時に保存された `norm_params.npz` の値と一致していることを確認：

```python
# norm_params.npz の値を確認
import numpy as np
params = np.load("log/<tag>/norm_params.npz")
print("data_min:", params["data_min"])
print("data_max:", params["data_max"])
```

この値を `target_publisher_convTAiRNN_v2.py` 内の `data_min_v2`, `data_max_v2` に手動で設定する。

#### (3) モデルの読み込み

スクリプト内でチェックポイントのパスを指定して実行する。
使用するエポックは TensorBoard のテストロスが最も低い時点のチェックポイントを選ぶ。

### ΔPos の累積について

モデルは絶対位置ではなく**1ステップのΔPos（差分）**を出力する。
テストスクリプトでは：
```python
y_joint[:, :3] += joint_t[:, :3]   # ΔPosを現在位置に足して次の絶対位置を計算
```
としている。ロボット制御スクリプトでも同様の処理が必要。

---

## よくあるエラーと対処

| エラー | 原因 | 対処 |
|-------|------|------|
| `checkpoint not found` | `--ckpt` のパスが間違っている | `log/<tag>/` 以下のファイル名を確認 |
| `norm_params.npz not found` | テスト前に学習を実行していない | 学習を先に実行する |
| `FileNotFoundError: dataset_root_dir` | データパスが存在しない | スクリプト内の `dataset_root_dir` を修正 |
| `CUDA out of memory` | VRAM不足 | `--batch_size` を減らす / `--im_size 64` に下げる |
| 損失が NaN になる | 学習率が高すぎる / データの値が異常 | `--lr 1e-5` に下げる / `norm_params.npz` の値を確認 |
