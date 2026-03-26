import math
from pathlib import Path  # 変更: osモジュールの代わりにpathlibを使用

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # visualize_overlay_and_save で使用
from ModelCode.layer import InverseSpatialSoftmax

# 変更: pos予測ヘッドの入力にdec_ptsを追加


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # 入力ゲート、忘却ゲート、セルゲート、出力ゲートをまとめて畳み込みで計算
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 入力と隠れ状態を結合 (Channel方向にConcat)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (B, input+hidden, H, W)
        combined_conv = self.conv(combined)

        # 4つのゲートに分割
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 状態更新
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        )


def create_position_encoding(width: int, height: int):
    pos_x, pos_y = np.meshgrid(np.linspace(0.0, 1.0, width), np.linspace(0.0, 1.0, height), indexing="xy")

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


class ConvTAiRNNv3(nn.Module):
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
        attn_dim: int = 64,
        n_heads: int = 1,  # MHA head 数
        visualize_on_forward: bool = False,
        folder_name: str = "noname",
        log_name: str = "noname",
    ):
        super(ConvTAiRNNv3, self).__init__()

        self.visualize_on_forward = visualize_on_forward
        self.current_vis_output_dir = None
        if self.visualize_on_forward:
            default_dir = Path("./output") / log_name / folder_name
            self.current_vis_output_dir = str(default_dir)
            Path(self.current_vis_output_dir).mkdir(parents=True, exist_ok=True)

        self.n_heads = n_heads
        self.k_dim = k_dim
        self.joint_dim = joint_dim
        self.attn_dim = attn_dim
        self.temperature = temperature
        self.rec_dim = rec_dim
        activation = nn.LeakyReLU(negative_slope=0.3)

        self.h0_from_state = nn.Sequential(
            nn.Linear(joint_dim, rec_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(rec_dim, rec_dim),
        )

        self.c0_from_state = nn.Sequential(
            nn.Linear(joint_dim, rec_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(rec_dim, rec_dim),
        )

        # 画像encoder
        self.im_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size, 1, 0),
            activation,
            nn.Conv2d(16, 32, kernel_size, 1, 0),
            activation,
            nn.Conv2d(32, k_dim, kernel_size, 1, 0),  # k_dimチャンネル
            activation,
        )

        self.token_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(64),
            activation,
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(128),
            activation,
            nn.Conv2d(128, attn_dim, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(attn_dim),
            activation,
        )

        with torch.no_grad():
            Hp, Wp = self.token_encoder(torch.zeros(1, 3, *im_size)).shape[-2:]
        self.Hp, self.Wp = Hp, Wp

        self.pos_encoder = LearnablePositionalEncoding(Hp, Wp, attn_dim)

        pos_x, pos_y = create_position_encoding(Hp, Wp)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        # Cross-Attention
        self.ln_q = nn.LayerNorm(attn_dim)
        self.ln_kv = nn.LayerNorm(attn_dim)
        self.mha = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=n_heads, batch_first=True, dropout=0.1)
        self.ln_out = nn.LayerNorm(attn_dim)

        # 2. 状態からコンテキストを作るための独立したプロジェクション層を定義
        self.object_queries = nn.Parameter(torch.zeros(1, self.k_dim, self.attn_dim))
        nn.init.xavier_uniform_(self.object_queries)

        hidden_dim = 32  # or 64 程度
        self.state_to_ctx = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, self.k_dim * self.attn_dim),
        )

        # RNN
        conv_input_dim = self.k_dim + self.attn_dim
        self.rec = ConvLSTMCell(input_dim=conv_input_dim, hidden_dim=self.rec_dim, kernel_size=3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Decoders
        self.decoder_pos = nn.Sequential(
            nn.Linear(self.rec_dim, rec_dim // 2),
            activation,
            nn.Dropout(0.1),
            nn.Linear(self.rec_dim // 2, rec_dim // 4),
            activation,
            nn.Dropout(0.1),
            nn.Linear(self.rec_dim // 4, 3),
        )
        self.decoder_g = nn.Sequential(
            nn.Linear(self.rec_dim, rec_dim // 4),
            activation,
            nn.Dropout(0.2),  # 変更
            nn.Linear(rec_dim // 4, 1),
        )

        self.decoder_point = nn.Sequential(
            nn.Linear(rec_dim, rec_dim // 2),
            activation,
            nn.Linear(rec_dim // 2, rec_dim // 4),
            activation,
            nn.Linear(rec_dim // 4, k_dim * 2),
        )
        self.decoder_joint = nn.Sequential(
            nn.Linear(self.k_dim * rec_dim, rec_dim // 2),
            activation,
            nn.Linear(rec_dim // 2, rec_dim // 4),
            activation,
            nn.Linear(rec_dim // 4, joint_dim),
        )

        self.map_predictor = nn.Sequential(
            nn.Conv2d(rec_dim, 32, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, k_dim, 1, 1, 0),  # 1x1 ConvでKチャンネルに
            nn.Sigmoid(),  # 0.0~1.0のヒートマップにする
        )

        self.decoder_image = nn.Sequential(
            nn.ConvTranspose2d(k_dim, 32, kernel_size, 1, 0),
            activation,
            nn.ConvTranspose2d(32, 16, kernel_size, 1, 0),
            activation,
            nn.ConvTranspose2d(16, 3, kernel_size, 1, 0),
        )

        self.apply(self._weights_init)
        self.vis_cnt = 0

        # 追加
        with torch.no_grad():
            # logit(p) = log(p/(1-p))
            bias_init = math.log(0.5 / max(1e-8, 1.0 - 0.5))
            self.decoder_g[-1].bias.fill_(bias_init)

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

    def forward(self, xi, xv, state=None):
        """
        xi: (B,C,H,W) 画像,   xv: (B,joint_dim) 関節角
        戻り値は従来と同じ: (y_img, y_joint, enc_pts, dec_pts, rnn_state)
        """
        B = xi.size(0)
        xv_joint, xv_extra = torch.split(xv, [self.joint_dim, 3], dim=1)

        # 1) 画像 → token (K/V)
        tok = self.token_encoder(xi)  # (B, attn_dim, H', W')
        KV = tok.flatten(2).transpose(1, 2)  # (B, N, attn_dim)

        KV_with_pos = self.pos_encoder(KV)  # (B, N, attn_dim)
        KV = self.ln_kv(KV_with_pos)
        preLN_KV = KV.transpose(1, 2).view(B, self.attn_dim, self.Hp, self.Wp)

        # 2) 関節角 → Query
        concept_queries = self.object_queries.expand(B, -1, -1)  # (B,K,D)

        ctx = self.state_to_ctx(xv_extra)  # (B, K*D)
        state_context = ctx.view(B, self.k_dim, self.attn_dim)  # (B,K,D)

        Q = concept_queries + state_context  # (B,K,D)
        Q = self.ln_q(Q)

        # 3) Cross-Attention
        attn_out, attn_w = self.mha(
            Q, KV, KV, need_weights=True, average_attn_weights=False
        )  # attn_w: (B, n_heads, k_dim, N)
        attn_out = self.ln_out(attn_out + Q)  # residual + LN

        enc_w = attn_w.mean(dim=1)  # (B, K, N)

        # 温度をかけたいならここで（任意）
        if self.temperature is not None:
            log_w = torch.log(enc_w + 1e-8) / self.temperature
            enc_w = F.softmax(log_w, dim=-1)  # (B, K, N)

        enc_map = enc_w.view(B, self.k_dim, self.Hp, self.Wp)  # (B,K,Hp,Wp)
        # self.attn_maps = enc_map.detach()  # rp_loss 用

        processed_attn_w = attn_w.flatten(1, 2)
        attn_map = processed_attn_w.view(B, self.n_heads * self.k_dim, self.Hp, self.Wp)
        eps = 1e-8
        attn_log = torch.log(processed_attn_w + eps) / self.temperature
        attn_sharp = F.softmax(attn_log, dim=-1)
        self.attn_maps = attn_sharp

        # 5) RNN ―――
        # lstm_in = attn_map
        lstm_in = torch.cat([attn_map, tok], dim=1)

        if state is None:
            h0 = self.h0_from_state(xv_joint)  # (B, rec_dim)
            c0 = self.c0_from_state(xv_joint)  # (B, rec_dim)
            h0 = h0.view(B, self.rec_dim, 1, 1).expand(-1, -1, self.Hp, self.Wp)
            c0 = c0.view(B, self.rec_dim, 1, 1).expand(-1, -1, self.Hp, self.Wp)
            state = (h0, c0)
        h, c = self.rec(lstm_in, state)

        # 6) 画像復元 ―――
        im_hid = self.im_encoder(xi)
        pred_attn_map = self.map_predictor(h)
        weighted_feature = im_hid * pred_attn_map  # skip connection + LSTM state
        y_image = self.decoder_image(weighted_feature)

        # 状態推定
        map_sums = pred_attn_map.sum(dim=(2, 3), keepdim=True) + 1e-8
        normalized_maps = pred_attn_map / map_sums
        slot_features = torch.einsum("bkhw, bchw -> bkc", normalized_maps, h)
        decoder_input = slot_features.reshape(B, self.k_dim * self.rec_dim)

        y_joint = self.decoder_joint(decoder_input)

        # 変更箇所
        if self.visualize_on_forward:
            self._visualize_step(xi, tok, preLN_KV, processed_attn_w, attn_sharp)

        # 8) 戻り値  ※dec_pts_px を flatten して返す
        return y_image, y_joint, enc_map, pred_attn_map, (h, c)

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

        # # --- 1. Attention Matrix (Query vs. Token) のヒートマップ ---
        # plt.figure(figsize=(20, 2))  # サイズを現実的に調整
        # plt.imshow(attn_vis.detach().cpu().numpy(), aspect="auto", cmap="jet")
        # plt.colorbar(label="Attention Weight")
        # plt.xlabel("Token Index")
        # plt.ylabel("Query Index")
        # plt.title(f"Attention Matrix (Step {step})")
        # plt.tight_layout()
        # plt.savefig(output_dir / f"attn_matrix_step{step}.png")
        # plt.close()

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

        # # --- 4. Token Encoder の特徴マップ ---
        # save_path_token = output_dir / f"token_encoder_features_step{step}.png"
        # visualize_feature_maps_and_save(tok_vis, save_path_token, map_type_name="TokenEnc")

        # # --- 5. Key/Value の特徴マップ ---
        # save_path_kv = output_dir / f"kv_features_step{step}.png"
        # visualize_feature_maps_and_save(kv_vis, save_path_kv, map_type_name="KV")

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
