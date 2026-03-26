import datetime  # タイムスタンプのために追加
import math
import os
import re
from pathlib import Path  # 変更: osモジュールの代わりにpathlibを使用

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # visualize_overlay_and_save で使用
from ModelCode.layer import InverseSpatialSoftmax

# 変更: 自身の予測結果を次の入力にする


def create_position_encoding(width: int, height: int):
    pos_x, pos_y = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="xy")

    pos_x = torch.from_numpy(pos_x.reshape(height * width)).float()
    pos_y = torch.from_numpy(pos_y.reshape(height * width)).float()

    return pos_x, pos_y


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, Hp, Wp, embed_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.num_patches = Hp * Wp
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.001)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 入力テンソル。形状は (B, N, C) または (B, C, H, W) を想定。
                              (B, N, C) の場合は N = Hp * Wp, C = embed_dim
                              (B, C, H, W) の場合は flatten して N にする必要がある。
                              ここでは (B, N, C) を想定して加算する。
        Returns:
            torch.Tensor: 位置エンコーディングが加算されたテンソル。
        """
        # self.pos_embed は (1, N, C) なので、バッチ次元 B に合わせてブロードキャストされる
        return x + self.pos_embed


class MultiTAiRNN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        rec_dim: int,  # LSTMcellのrecurrent次元
        k_dim: int = 5,
        joint_dim: int = 7,  # ロボットの状態次元
        temperature: float = 1e-4,
        heatmap_size: float = 0.1,
        kernel_size: int = 3,
        im_size: list = [64, 64],
        attn_dim: int = 64,  # cross-attention 埋め込み次元
        n_heads: int = 1,  # MHA head 数
        visualize_on_forward: bool = False,
        test_name: str = "noname",
        input_mode: str = "real",
        pred_len: int = 1,
    ):
        super(MultiTAiRNN, self).__init__()

        self.input_mode = input_mode
        self.pred_len = pred_len

        self.visualize_on_forward = visualize_on_forward
        if self.visualize_on_forward:
            base_vis_output_dir = "/work/gn45/n45002/TAiRNN/output/"
            self.current_vis_output_dir = os.path.join(base_vis_output_dir, timestamp)
            os.makedirs(self.current_vis_output_dir, exist_ok=True)

            match = re.search(r"(\d{8}_\d{4}_\d{2})", test_name)
            if match:
                date_str = match.group(1)
                txt_file_path = os.path.join(self.current_vis_output_dir, f"{date_str}.txt")
                with open(txt_file_path, "w") as f:
                    f.write("")
            else:
                print("Error: test_name から日付文字列を抽出できませんでした。")

        self.n_heads = n_heads
        self.k_dim = k_dim
        self.joint_dim = joint_dim
        self.attn_dim = attn_dim
        self.temperature = temperature
        activation = nn.LeakyReLU(negative_slope=0.3)

        # 画像encoder
        # 画像encoder
        self.im_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),
            activation,
            nn.Conv2d(32, k_dim, 3, 1, 0),  # k_dimチャンネル
            activation,
        )

        self.token_encoder = nn.Sequential(
            # 第1ブロック: 空間情報を保持しつつ初期特徴を抽出
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),  # 正規化層を追加
            activation,
            # 第2ブロック: チャネル数を増やし、より複雑な特徴へ
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            activation,
            # 第3ブロック: 最終的なattn_dimへ
            nn.Conv2d(128, attn_dim, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(attn_dim),
            activation,
            # 必要であればさらに層を追加したり、プーリング層を挟むことも検討
        )

        with torch.no_grad():
            Hp, Wp = self.token_encoder(torch.zeros(1, 3, *im_size)).shape[-2:]
        self.Hp, self.Wp = Hp, Wp

        self.pos_encoder = LearnablePositionalEncoding(Hp, Wp, attn_dim)

        pos_x, pos_y = create_position_encoding(Hp, Wp)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        # Cross-Attention
        self.q_proj = nn.Linear(joint_dim, k_dim * attn_dim, bias=False)
        self.q_proj_v2 = nn.Linear(3, k_dim * attn_dim, bias=False)
        self.ln_q = nn.LayerNorm(attn_dim)
        self.ln_kv = nn.LayerNorm(attn_dim)
        self.mha = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=n_heads, batch_first=True, dropout=0.1)
        self.ln_out = nn.LayerNorm(attn_dim)

        # 1. 役割を担う「概念クエリ」を定義
        self.object_queries = nn.Parameter(torch.zeros(1, self.k_dim, self.attn_dim))
        nn.init.xavier_uniform_(self.object_queries)
        # 2. 状態からコンテキストを作るための独立したプロジェクション層を定義
        self.context_projections = nn.ModuleList([nn.Linear(3, self.attn_dim, bias=False) for _ in range(self.k_dim)])

        # RNN
        self.rec = nn.LSTMCell(joint_dim + k_dim * 2, rec_dim)

        # Decoders

        self.decoder_joint = nn.Sequential(nn.Linear(rec_dim, joint_dim), activation)
        self.decoder_point = nn.Sequential(nn.Linear(rec_dim, k_dim * 2), activation)

        self.issm = InverseSpatialSoftmax(width=Hp, height=Wp, heatmap_size=heatmap_size, normalized=True)

        # TODO:ResNetを削除したため、パディングなどを変更
        self.decoder_image = nn.Sequential(
            nn.ConvTranspose2d(k_dim, 32, 3, 1, 0),
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 0),
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 0),
            activation,
        )

        self.apply(self._weights_init)
        self.vis_cnt = 0

    def _weights_init(self, m):
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, xi, xv, state=None, pred_len=1, input_mode="real"):
        """
        :param xi: (B,C,H,W) 画像
        :param xv: (B,joint_dim+3) 関節角 + extra
        :param state: (h,c) LSTMの内部状態
        :param pred_len: 予測するステップ数 (デフォルトは1)
        :param input_mode: 'real' or 'pred'
             'real': 外部からの入力xi, xvを使って1ステップ予測（従来の動作）
             'pred': 自身の予測結果を次の入力として、pred_lenステップ予測
        :return: 各予測結果をリストとして返す
        """

        # 予測結果を格納するリスト
        y_image_list, y_joint_list = [], []
        enc_pts_list, dec_pts_list = [], []

        # 最初の入力は常に外部から与えられる
        current_xi = xi
        current_xv = xv

        for t in range(pred_len):
            B = current_xi.size(0)
            xv_joint, xv_extra = torch.split(current_xv, [self.joint_dim, 3], dim=1)

            # 1) 画像 → token (K/V)
            tok = self.token_encoder(current_xi)
            KV = tok.flatten(2).transpose(1, 2)
            KV_with_pos = self.pos_encoder(KV)
            KV = self.ln_kv(KV_with_pos)
            preLN_KV = KV.transpose(1, 2).view(B, self.attn_dim, self.Hp, self.Wp)

            # 2) 関節角 → Query
            concept_queries = self.object_queries.expand(B, -1, -1)
            context_vectors = []
            for i in range(self.k_dim):
                ctx_i = self.context_projections[i](xv_extra)
                context_vectors.append(ctx_i)
            state_context = torch.stack(context_vectors, dim=1)
            Q = concept_queries + state_context
            Q = self.ln_q(Q)

            # 3) Cross-Attention
            attn_out, attn_w = self.mha(Q, KV, KV, need_weights=True, average_attn_weights=False)
            attn_out = self.ln_out(attn_out + Q)

            # 4) attention → (x,y) 座標に変換
            processed_attn_w = attn_w.squeeze(1)
            eps = 1e-8
            attn_log = torch.log(processed_attn_w + eps) / self.temperature
            attn_sharp = F.softmax(attn_log, dim=-1)
            expected_x = torch.sum(self.pos_x * attn_sharp, dim=-1, keepdim=True)
            expected_y = torch.sum(self.pos_y * attn_sharp, dim=-1, keepdim=True)
            keys = torch.cat([expected_x, expected_y], -1)
            enc_pts = keys.reshape(B, -1)  # (B, k_dim * 2)

            # 5) RNN
            hid_in = torch.cat([enc_pts, xv_joint], -1)
            h, c = self.rec(hid_in, state)
            state = (h, c)  # 次のループのためにstateを更新

            # 6) Decoders
            y_joint = self.decoder_joint(h)
            dec_pts = self.decoder_point(h)

            # --- 画像再構成 ---
            # 'pred'モードでは、予測した画像y_imageを次の入力xiとする
            # そのため、y_imageの計算を先に行う
            im_hid = self.im_encoder(current_xi)
            dec_pts_in = dec_pts.reshape(-1, self.k_dim, 2)
            heatmap = self.issm(dec_pts_in)
            y_image = self.decoder_image(heatmap * im_hid)

            # --- 予測結果をリストに保存 ---
            y_image_list.append(y_image)
            y_joint_list.append(y_joint)
            enc_pts_list.append(enc_pts)
            dec_pts_list.append(dec_pts)

            # 'pred'モードの場合、次の入力として自身の予測値を使う
            if input_mode == "pred":
                current_xi = y_image
                # 次のxvを生成 (y_jointから必要な部分を抽出)
                # 注: xv_extra (3次元) の部分をどう扱うかは設計による
                # ここでは、予測した関節角度と、前のxv_extraを結合すると仮定
                next_xv_joint = y_joint[:, : self.joint_dim]
                current_xv = torch.cat([next_xv_joint, xv_extra], dim=1)

        # 変更箇所
        if self.visualize_on_forward:
            self._visualize_step(xi, tok, preLN_KV, processed_attn_w, attn_sharp)

        # 8) 戻り値  ※dec_pts_px を flatten して返す
        return y_image_list, y_joint_list, enc_pts_list, dec_pts_list, state

    def _visualize_step(self, input_image, token_features, kv_features, attn_weights, sharp_attn_weights):
        """
        学習中の特定ステップにおける中間特徴量やAttentionを可視化し、保存する。
        元のforwardメソッドにあった可視化ロジックをここに集約。
        """
        # --- 準備: 可視化対象のテンソル (バッチの最初の要素) と保存先ディレクトリを取得 ---
        img_vis = input_image[0]
        tok_vis = token_features[0]
        kv_vis = kv_features[0]
        attn_vis = attn_weights[0]  # 温度適用前
        sharp_attn_vis = sharp_attn_weights[0]  # 温度適用後

        original_H, original_W = img_vis.shape[-2:]
        output_dir = Path(self.current_vis_output_dir)
        step = self.vis_cnt

        # --- 1. Attention Matrix (Query vs. Token) のヒートマップ ---
        plt.figure(figsize=(20, 2))  # サイズを現実的に調整
        plt.imshow(attn_vis.detach().cpu().numpy(), aspect="auto", cmap="jet")
        plt.colorbar(label="Attention Weight")
        plt.xlabel("Token Index")
        plt.ylabel("Query Index")
        plt.title(f"Attention Matrix (Step {step})")
        plt.tight_layout()
        plt.savefig(output_dir / f"attn_matrix_step{step}.png")
        plt.close()

        # --- 2. 温度適用前のAttentionヒートマップ (k_dimごと) ---
        pre_temp_maps = attn_vis.reshape(self.k_dim, self.Hp, self.Wp)
        save_path_pre = output_dir / f"attn_heatmap_pre_temp_step{step}.png"
        visualize_attention_heatmaps(pre_temp_maps, save_path_pre, title_prefix="Pre-Temp Attn")

        # --- 3. 温度適用後のAttentionと元画像のオーバーレイ ---
        post_temp_maps = sharp_attn_vis.reshape(self.k_dim, self.Hp, self.Wp)
        save_path_overlay = output_dir / f"attn_overlay_step{step}.png"
        aspect_ratio = original_W / original_H if original_H > 0 else 1
        visualize_all_k_attn_overlays(
            img_vis,
            post_temp_maps,
            (original_H, original_W),
            save_path_overlay,
            self.k_dim,
            subplot_figsize=(4 * aspect_ratio, 4),
        )

        # --- 4. Token Encoder の特徴マップ ---
        save_path_token = output_dir / f"token_encoder_features_step{step}.png"
        visualize_feature_maps_and_save(tok_vis, save_path_token, map_type_name="TokenEnc")

        # --- 5. Key/Value の特徴マップ ---
        save_path_kv = output_dir / f"kv_features_step{step}.png"
        visualize_feature_maps_and_save(kv_vis, save_path_kv, map_type_name="KV")

        self.vis_cnt += 1


def visualize_all_k_attn_overlays(
    input_image_tensor,  # (C, H_orig, W_orig) 形式の元画像テンソル (例: xi[0])
    all_k_attn_maps_tensor,  # (k_dim, Hp, Wp) 形式の全k_dimのアテンションマップ
    original_image_size,  # (H_orig, W_orig) タプル
    save_path,
    k_dim,
    subplot_figsize=(4, 4),  # 各サブプロットの推奨サイズ (インチ)
    alpha=0.6,
    cmap="jet",
):
    """
    入力画像に各k_dimのアテンションマップを重ね合わせ、横に並べて1枚の画像として保存する。
    """
    if input_image_tensor is None or all_k_attn_maps_tensor is None:
        print("Input image or attention maps tensor is None.")
        return
    if all_k_attn_maps_tensor.shape[0] != k_dim:
        print(
            f"Mismatch between k_dim ({k_dim}) and all_k_attn_maps_tensor.shape[0] ({all_k_attn_maps_tensor.shape[0]})"
        )
        return

    H_orig, W_orig = original_image_size

    # 1. 入力画像の準備 (一度だけ行う)
    img_to_show_base = input_image_tensor.detach().cpu()
    if img_to_show_base.shape[0] == 3:  # カラー画像
        img_to_show_base_np = img_to_show_base.permute(1, 2, 0).numpy()
    elif img_to_show_base.shape[0] == 1:  # グレースケール画像
        img_to_show_base_np = img_to_show_base.squeeze().numpy()
    else:
        print(f"Unsupported input image tensor shape: {input_image_tensor.shape}")
        return
    img_to_show_base_np = np.clip(img_to_show_base_np, 0, 1)

    # 2. サブプロットの設定
    num_cols = k_dim
    num_rows = 1

    # 図全体のサイズを動的に計算

    fig_width = num_cols * subplot_figsize[0]
    fig_height = num_rows * subplot_figsize[1]

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False)
    # squeeze=Falseにより、num_rows=1またはnum_cols=1でも常に2D配列でaxesが返る

    for k_idx in range(k_dim):
        ax = axes[0, k_idx]  # 1行なので axes[0, k_idx]

        # 3. アテンションマップの準備とリサイズ
        single_attn_map = all_k_attn_maps_tensor[k_idx].detach().cpu()  # (Hp, Wp)
        map_resized = (
            F.interpolate(
                single_attn_map.unsqueeze(0).unsqueeze(0), size=(H_orig, W_orig), mode="bilinear", align_corners=False
            )
            .squeeze()
            .numpy()
        )

        # 4. 重ね合わせ表示
        ax.imshow(img_to_show_base_np)
        im = ax.imshow(map_resized, cmap=cmap, alpha=alpha)
        ax.set_title(f"Overlay Attn (k={k_idx})")
        ax.axis("off")

    fig.tight_layout(pad=0.5)  # サブプロット間の空白や全体の余白を調整

    # 5. 保存
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Failed to save all k-dim attention overlays: {e}")
    plt.close(fig)


def visualize_feature_maps_and_save(
    feature_maps_tensor,  # (num_maps, H_feat, W_feat) 形式のテンソル (例: tok[0] や im_hid[0])
    save_path,
    num_maps_to_show=16,  # 表示するマップの最大数
    map_type_name="Feature",  # タイトルやファイル名用 (例: "TokenEncoder", "ImageEncoder")
    subplot_size=(2, 2),  # 各サブプロットのインチ単位のサイズ (幅, 高さ)
    max_cols=4,  # 1行に表示するサブプロットの最大数
    cmap="viridis",
):
    """
    特徴マップの各チャネルをヒートマップとして可視化し、指定されたパスに保存する。
    画面には表示しない。
    """
    if feature_maps_tensor is None:
        print(f"{map_type_name} maps tensor is None. Cannot visualize.")
        return

    feature_maps_cpu = feature_maps_tensor.detach().cpu()  # (num_maps, H_feat, W_feat)

    num_actual_maps = feature_maps_cpu.shape[0]
    H_feat = feature_maps_cpu.shape[1]
    W_feat = feature_maps_cpu.shape[2]

    num_plots = min(num_actual_maps, num_maps_to_show)
    if num_plots == 0:
        print(f"No {map_type_name} maps to show.")
        return

    num_vis_cols = min(num_plots, max_cols)
    num_vis_rows = math.ceil(num_plots / num_vis_cols)

    fig_width = num_vis_cols * subplot_size[0]
    fig_height = num_vis_rows * subplot_size[1]

    fig, axes = plt.subplots(num_vis_rows, num_vis_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for i in range(num_plots):
        ax = axes[i]
        channel_map = feature_maps_cpu[i]
        im = ax.imshow(channel_map.numpy(), cmap=cmap, aspect="auto")  # aspect='auto' を追加
        ax.set_title(f"{map_type_name} Ch: {i}")
        ax.axis("off")
        # カラーバーは数が多すぎると見づらくなるため、必要に応じてコメント解除または調整
        # fig.colorbar(im, ax=ax, shrink=0.6)

    # 使わない余分なサブプロットを非表示にする
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(pad=0.5)
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Failed to save {map_type_name} maps: {e}")
    plt.close(fig)  # 画面表示せず閉じる


def visualize_attention_heatmaps(
    attn_maps_tensor, save_path, title_prefix="Attn", cmap="jet"  # torch.Tensor, 形状 (k_dim, Hp, Wp)
):
    """
    attn_maps_tensor の各 k_dim チャネルをヒートマップとして並べ、1枚の画像にまとめて保存する。

    attn_maps_tensor: (k_dim, Hp, Wp)
    """
    import matplotlib.pyplot as plt

    k_dim, Hp, Wp = attn_maps_tensor.shape
    num_cols = min(k_dim, 4)  # 一度に最大 4 列まで
    num_rows = (k_dim + num_cols - 1) // num_cols

    fig_width = num_cols * 3
    fig_height = num_rows * 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False)

    attn_cpu = attn_maps_tensor.detach().cpu().numpy()

    for idx in range(k_dim):
        r = idx // num_cols
        c = idx % num_cols
        ax = axes[r][c]
        im = ax.imshow(attn_cpu[idx], cmap=cmap, aspect="auto")
        ax.set_title(f"{title_prefix} k={idx}")
        ax.axis("off")

    # 余ったサブプロットは消す
    for j in range(k_dim, num_rows * num_cols):
        r = j // num_cols
        c = j % num_cols
        fig.delaxes(axes[r][c])

    plt.tight_layout(pad=0.3)
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Failed to save {title_prefix} heatmaps: {e}")
    plt.close(fig)
