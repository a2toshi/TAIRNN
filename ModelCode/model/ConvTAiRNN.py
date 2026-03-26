import math
import os
from pathlib import Path  # 変更: osモジュールの代わりにpathlibを使用

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # 入力と隠れ状態を結合したものを一度に畳み込む
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # 4つのゲート分
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        """
        Args:
            input_tensor (torch.Tensor): (B, C_in, H, W) の入力テンソル。
            cur_state (tuple): (h_cur, c_cur) のタプル。
                h_cur (torch.Tensor): (B, C_hid, H, W) の前の隠れ状態。
                c_cur (torch.Tensor): (B, C_hid, H, W) の前のセル状態。

        Returns:
            tuple: (h_next, c_next) のタプル。
        """
        h_cur, c_cur = cur_state

        # 入力と隠れ状態をチャネル方向に結合
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # 畳み込みを一回で実行
        combined_conv = self.conv(combined)

        # ゲートごとに分割 (input, forget, output, cell)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # ゲートの活性化関数
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 次のセル状態と隠れ状態を計算
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        隠れ状態とセル状態をゼロで初期化するヘルパー関数。
        Args:
            batch_size (int): バッチサイズ。
            image_size (tuple): 特徴マップのサイズ (H, W)。
        """
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        )


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


class ConvTAiRNN(nn.Module):
    """
    TAiRNNの構造をベースに、LSTMCellをConvLSTMCellに置き換えたモデル。
    Attention Mapを直接、空間情報を保持したまま時系列処理に流す。
    """

    def __init__(
        self,
        rec_dim: int,  # ConvLSTMの隠れ層チャネル数
        k_dim: int = 5,
        joint_dim: int = 7,  # ロボットの状態次元
        temperature: float = 1e-4,
        kernel_size: int = 3,
        im_size: list = [64, 64],
        attn_dim: int = 64,  # cross-attention 埋め込み次元
        n_heads: int = 2,  # MHA head 数
        state_embed_dim: int = 16,
        visualize_on_forward: bool = False,
        output_dir: str = "noname",
    ):
        super(ConvTAiRNN, self).__init__()

        self.visualize_on_forward = visualize_on_forward
        self.current_vis_output_dir = None
        if self.visualize_on_forward:
            if output_dir == "noname":
                raise ValueError("visualize_on_forward is True, but output_dir was not provided.")
            self.current_vis_output_dir = output_dir
            os.makedirs(self.current_vis_output_dir, exist_ok=True)

        self.n_heads = n_heads
        self.k_dim = k_dim
        self.joint_dim = joint_dim
        self.attn_dim = attn_dim
        self.temperature = temperature
        self.state_embed_dim = state_embed_dim

        activation = nn.LeakyReLU(negative_slope=0.3)

        # 画像encoder
        # self.im_encoder = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 1, 0),
        #     activation,
        #     nn.Conv2d(16, 32, 3, 1, 0),
        #     activation,
        #     nn.Conv2d(32, k_dim, 3, 1, 0),  # k_dimチャンネル
        #     activation,
        # )
        self.im_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),
            activation,
            nn.Conv2d(32, self.n_heads * self.k_dim, 3, 1, 0),  # k_dim を変更
            activation,
        )

        self.token_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),  # 正規化層を追加
            activation,
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            activation,
            nn.Conv2d(64, attn_dim, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(attn_dim),
            activation,
        )

        with torch.no_grad():
            Hp, Wp = self.token_encoder(torch.zeros(1, 3, *im_size)).shape[-2:]
        self.Hp, self.Wp = Hp, Wp

        self.pos_encoder = LearnablePositionalEncoding(Hp, Wp, attn_dim)

        # Cross-Attention
        self.q_proj = nn.Linear(joint_dim, k_dim * attn_dim, bias=False)
        self.q_proj_v2 = nn.Linear(3, k_dim * attn_dim, bias=False)
        self.ln_q = nn.LayerNorm(attn_dim)
        self.ln_kv = nn.LayerNorm(attn_dim)
        self.mha = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=n_heads, batch_first=True, dropout=0.1)
        self.ln_out = nn.LayerNorm(attn_dim)

        self.object_queries = nn.Parameter(torch.zeros(1, self.k_dim, self.attn_dim))
        nn.init.xavier_uniform_(self.object_queries)
        self.context_projections = nn.ModuleList([nn.Linear(3, self.attn_dim, bias=False) for _ in range(self.k_dim)])

        # --- RNN部分の変更点 ---
        # 1. ロボット状態を空間特徴マップに埋め込むためのプロジェクタ
        self.state_projector = nn.Linear(joint_dim, self.state_embed_dim)

        # 2. ConvLSTMCellの定義
        # convlstm_input_dim = self.k_dim + self.state_embed_dim
        # convlstm_input_dim = self.k_dim + self.attn_dim + self.state_embed_dim
        convlstm_input_dim = (self.n_heads * self.k_dim) + self.attn_dim + self.state_embed_dim
        self.conv_lstm = ConvLSTMCell(
            input_dim=convlstm_input_dim,
            hidden_dim=rec_dim,  # rec_dimを隠れ状態のチャネル数として使用
            kernel_size=(kernel_size, kernel_size),
            bias=True,
        )

        # 3. ConvLSTMの出力をデコーダ用のベクトルに変換する層
        self.decoder_input_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(rec_dim, rec_dim), activation  # 必要に応じて次元調整
        )

        # Decoders
        self.decoder_joint = nn.Sequential(nn.Linear(rec_dim, joint_dim), activation)
        # self.decoder_map = nn.Conv2d(in_channels=rec_dim, out_channels=self.k_dim, kernel_size=1)
        # self.decoder_map = nn.Conv2d(in_channels=rec_dim, out_channels=self.n_heads * self.k_dim, kernel_size=1)
        self.decoder_map = nn.Sequential(
            nn.Conv2d(in_channels=rec_dim, out_channels=rec_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(in_channels=rec_dim, out_channels=self.n_heads * self.k_dim, kernel_size=1),
        )

        # self.decoder_image = nn.Sequential(
        #     nn.ConvTranspose2d(k_dim, 32, 3, 1, 0),
        #     activation,
        #     nn.ConvTranspose2d(32, 16, 3, 1, 0),
        #     activation,
        #     nn.ConvTranspose2d(16, 3, 3, 1, 0),
        #     activation,
        # )
        self.decoder_image = nn.Sequential(
            nn.ConvTranspose2d(self.n_heads * self.k_dim, 32, 3, 1, 0),  # k_dim を変更
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
        preLN_KV = KV.transpose(1, 2).view(B, self.attn_dim, self.Hp, self.Wp)  # debug

        # 2) 関節角 → Query
        concept_queries = self.object_queries.expand(B, -1, -1)
        context_vectors = []
        for i in range(self.k_dim):
            # xv_extra (B, 3) を、i番目のプロジェクション層で (B, attn_dim) に変換
            ctx_i = self.context_projections[i](xv_extra)
            context_vectors.append(ctx_i)
        # (B, k_dim, attn_dim) の形状にスタック
        state_context = torch.stack(context_vectors, dim=1)
        Q = concept_queries + state_context
        Q = self.ln_q(Q)

        # 3) Cross-Attention
        attn_out, attn_w = self.mha(
            Q, KV, KV, need_weights=True, average_attn_weights=False
        )  # attn_w: (B, n_heads, k_dim, N)
        attn_out = self.ln_out(attn_out + Q)  # residual + LN

        # 4) Attention Mapを空間的な形状に戻す
        # processed_attn_w = attn_w.squeeze(1)
        processed_attn_w = attn_w.flatten(1, 2)
        attn_map = processed_attn_w.view(B, self.n_heads * self.k_dim, self.Hp, self.Wp)
        # attn_map = processed_attn_w.view(B, self.k_dim, self.Hp, self.Wp)

        # 5) RNN (ConvLSTM)
        # 5a) ロボット状態を空間的にブロードキャスト
        projected_state = self.state_projector(xv_joint)  # (B, state_embed_dim)
        broadcasted_state = projected_state.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.Hp, self.Wp)

        # 5b) Attention Mapとロボット状態を結合してConvLSTMへの入力を作成
        conv_lstm_input = torch.cat([attn_map, broadcasted_state], dim=1)
        # conv_lstm_input = torch.cat([attn_map, tok, broadcasted_state], dim=1)

        # 5c) ConvLSTMセルを実行
        # stateがNoneの場合は初期状態を生成
        if state is None:
            state = self.conv_lstm.init_hidden(B, (self.Hp, self.Wp))

        h, c = self.conv_lstm(conv_lstm_input, state)

        # 6) デコーダのための隠れ状態を処理
        # h は (B, rec_dim, Hp, Wp) の特徴マップ
        decoder_input = self.decoder_input_proj(h)  # (B, rec_dim) のベクトルになる

        # 7) 次のロボット状態を予測
        y_joint = self.decoder_joint(decoder_input)

        # 8) 画像を復元
        pred_heatmap_raw = self.decoder_map(h)
        B, K, H, W = pred_heatmap_raw.shape
        heatmap_flat = pred_heatmap_raw.view(B, K, -1)
        # 温度付きソフトマックスを適用(TODO:変更)
        heatmap_logprob_flat = F.log_softmax(heatmap_flat / self.temperature, dim=-1)
        heatmap_logprob = heatmap_logprob_flat.view(B, K, H, W)

        predicted_heatmap = torch.exp(heatmap_logprob)
        im_hid = self.im_encoder(xi)
        y_image = self.decoder_image(predicted_heatmap * im_hid)

        # TODO: 可視化ロジックを新しいモデルの出力に合わせて修正する必要がある
        if self.visualize_on_forward:
            self._visualize_step(xi, preLN_KV, attn_map, predicted_heatmap)

        # 8) 戻り値
        return y_image, y_joint, attn_map, predicted_heatmap, (h, c)

    def _visualize_step(self, input_image, kv_features, input_attn_map, pred_heatmap):
        """
        学習中の特定ステップにおける中間特徴量やAttentionを可視化し、保存する。
        元のforwardメソッドにあった可視化ロジックをここに集約。
        """
        # --- 準備: 可視化対象のテンソル (バッチの最初の要素) と保存先ディレクトリを取得 ---
        img_vis = input_image[0]
        kv_vis = kv_features[0]
        input_attn_vis = input_attn_map[0]  # 温度適用前
        pred_heatmap_vis = pred_heatmap[0]  # 温度適用後

        output_dir = Path(self.current_vis_output_dir)
        step = self.vis_cnt

        # --- ① Keyの特徴マップ (`preLN_KV`) の可視化 ---
        save_path_kv = output_dir / f"step{step:04d}_kv_features.png"
        visualize_feature_maps_and_save(kv_vis, save_path_kv, map_type_name="KV Features")

        # --- ② Cross-AttentionのAttention Map (`attn_map`) の可視化 ---
        # 元画像と重ねて表示
        save_path_input_attn = output_dir / f"step{step:04d}_input_attn_overlay.png"
        visualize_all_k_attn_overlays(
            input_image_tensor=img_vis,
            all_k_attn_maps_tensor=input_attn_vis,
            original_image_size=img_vis.shape[-2:],
            save_path=save_path_input_attn,
            k_dim=self.k_dim * self.n_heads,
        )

        # --- ③ ConvLSTMから出力された予測Attention Map (`predicted_heatmap`) の可視化 ---
        # こちらも元画像と重ねて表示
        save_path_pred_heatmap = output_dir / f"step{step:04d}_pred_heatmap_overlay.png"
        visualize_all_k_attn_overlays(
            input_image_tensor=img_vis,
            all_k_attn_maps_tensor=pred_heatmap_vis,
            original_image_size=img_vis.shape[-2:],
            save_path=save_path_pred_heatmap,
            # k_dim=self.k_dim,
            k_dim=self.k_dim * self.n_heads,
        )

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
