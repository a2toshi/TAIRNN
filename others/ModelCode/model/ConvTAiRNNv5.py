import math
import os
from pathlib import Path  # 変更: osモジュールの代わりにpathlibを使用

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # visualize_overlay_and_save で使用

# 変更: convv4のpred_attn_mapを変更


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


class ConvTAiRNNv5(nn.Module):
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
        super(ConvTAiRNNv5, self).__init__()
        # print("メモ")

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
        conv_input_dim = self.joint_dim + self.k_dim + self.attn_dim
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
            # nn.Sigmoid(),  # 0.0~1.0のヒートマップにする
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
        B, _, Hp, Wp = tok.shape
        xv_map = xv_joint.view(B, self.joint_dim, 1, 1).expand(-1, -1, Hp, Wp)
        lstm_in = torch.cat([attn_map, tok, xv_map], dim=1)
        # lstm_in = torch.cat([torch.zeros_like(attn_map), tok, xv_map], dim=1) #アブレーション用

        if state is None:
            state = self.rec.init_hidden(B, (self.Hp, self.Wp))
        h, c = self.rec(lstm_in, state)

        # 6) 画像復元 ―――
        im_hid = self.im_encoder(xi)
        pred_logits = self.map_predictor(h)  # (B,K,Hp,Wp)
        pred_prob = torch.softmax(pred_logits.flatten(2), dim=-1).view(B, self.k_dim, self.Hp, self.Wp)
        # pred_attn_map = F.softmax(map_flat, dim=-1).view(B, K, H, W)
        N = self.Hp * self.Wp
        gate = torch.clamp(pred_prob * N, 0.0, 1.0)  # 期待値スケールを1付近に戻す
        weighted_feature = im_hid * gate
        y_image = self.decoder_image(weighted_feature)

        # 状態推定
        slot_features = torch.einsum("bkhw, bchw -> bkc", pred_prob, h)
        decoder_input = slot_features.reshape(B, self.k_dim * self.rec_dim)

        y_joint = self.decoder_joint(decoder_input)

        # 変更箇所
        if self.visualize_on_forward:
            self.visualize_attn_maps(
                xi=xi,
                attn_w=attn_w,  # (B,n_heads,K,N)
                enc_map=enc_map,  # (B,K,Hp,Wp)
                pred_attn_map=pred_prob,  # (B,K,Hp,Wp)
                save_np=False,
                save_png=True,
                max_batch_items=1,  # まずはb=0だけ
                save_every=1,  # 重いなら 5 や 10 に
            )

        # 8) 戻り値  ※dec_pts_px を flatten して返す
        return y_image, y_joint, enc_map, pred_prob, (h, c)

    def _to_uint8_rgb(self, x_chw: torch.Tensor) -> np.ndarray:
        """
        x_chw: (3,H,W) torch tensor on CPU/GPU
        戻り値: (H,W,3) uint8
        - 入力のスケールが [0,1] / [-1,1] / 任意の可能性があるので、可視化用にmin-max正規化
        """
        x = x_chw.detach().float().cpu()
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"Expected (3,H,W), got {tuple(x.shape)}")

        # min-max normalize per image for visualization robustness
        vmin = float(x.min())
        vmax = float(x.max())
        if vmax - vmin < 1e-8:
            x = torch.zeros_like(x)
        else:
            x = (x - vmin) / (vmax - vmin)

        x = (x.clamp(0, 1) * 255.0).byte()
        x = x.permute(1, 2, 0).numpy()  # HWC
        return x

    def _save_heatmap_png(self, path: str, heat: np.ndarray, title: str = "", vmin=0.0, vmax=1.0):
        """
        heat: (H,W) float numpy
        """
        plt.figure(figsize=(4, 4))
        plt.imshow(heat, vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _save_overlay_png(self, path: str, img_rgb: np.ndarray, heat: np.ndarray, title: str = "", alpha=0.45):
        """
        img_rgb: (H,W,3) uint8
        heat: (H,W) float
        """
        plt.figure(figsize=(4, 4))
        plt.imshow(img_rgb)
        plt.imshow(heat, alpha=alpha)  # colormapはmatplotlib default
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    @torch.no_grad()
    def visualize_attn_maps(
        self,
        xi: torch.Tensor,
        attn_w: torch.Tensor,  # (B,n_heads,k_dim,N)
        enc_map: torch.Tensor,  # (B,k_dim,Hp,Wp)
        pred_attn_map: torch.Tensor,  # (B,k_dim,Hp,Wp)
        save_np: bool = True,
        save_png: bool = True,
        max_batch_items: int = 1,
        save_every: int = 1,
    ):
        if not self.visualize_on_forward:
            return
        if self.current_vis_output_dir is None:
            return
        if (self.vis_cnt % max(1, save_every)) != 0:
            self.vis_cnt += 1
            return

        out_dir = self.current_vis_output_dir
        os.makedirs(out_dir, exist_ok=True)

        B = xi.shape[0]
        bmax = min(B, max_batch_items)

        # shape checks
        if attn_w.ndim != 4:
            raise ValueError(f"attn_w must be (B,n_heads,k_dim,N). got {tuple(attn_w.shape)}")
        if enc_map.ndim != 4 or pred_attn_map.ndim != 4:
            raise ValueError("enc_map/pred_attn_map must be (B,k_dim,Hp,Wp)")

        _, n_heads, K, N = attn_w.shape
        _, K2, Hp, Wp = enc_map.shape
        if K2 != K:
            raise ValueError(f"k_dim mismatch: attn_w K={K}, enc_map K={K2}")
        if N != Hp * Wp:
            raise ValueError(f"N != Hp*Wp. N={N}, Hp*Wp={Hp*Wp} -> cannot reshape attn_w to map")

        # (B,n_heads,K,Hp,Wp)
        attn_map_full = attn_w.detach().float().cpu().view(B, n_heads, K, Hp, Wp)
        enc_map_cpu = enc_map.detach().float().cpu()
        pred_map_cpu = pred_attn_map.detach().float().cpu()

        step_id = self.vis_cnt

        for b in range(bmax):
            sample_dir = os.path.join(out_dir, f"step_{step_id:06d}_b{b}")
            os.makedirs(sample_dir, exist_ok=True)

            img_rgb = self._to_uint8_rgb(xi[b])
            if save_png:
                plt.imsave(os.path.join(sample_dir, "input.png"), img_rgb)

            # --- npy保存（解析用） ---
            if save_np:
                np.save(
                    os.path.join(sample_dir, "attn_w.npy"), attn_w[b].detach().float().cpu().numpy()
                )  # (n_heads,K,N)
                np.save(os.path.join(sample_dir, "attn_map.npy"), attn_map_full[b].numpy())  # (n_heads,K,Hp,Wp)
                np.save(os.path.join(sample_dir, "enc_map.npy"), enc_map_cpu[b].numpy())  # (K,Hp,Wp)
                np.save(os.path.join(sample_dir, "pred_attn_map.npy"), pred_map_cpu[b].numpy())  # (K,Hp,Wp)

            if not save_png:
                continue

            # --- 1) attn_w -> map を head×K タイルで1枚にまとめる ---
            # flatten: (n_heads*K, Hp, Wp)
            attn_tiles = attn_map_full[b].reshape(n_heads * K, Hp, Wp).numpy()
            attn_titles = [f"h{h}_k{k}" for h in range(n_heads) for k in range(K)]
            self._save_heatmap_grid_png(
                os.path.join(sample_dir, "attn_w_grid.png"),
                attn_tiles,
                titles=attn_titles,
                suptitle="attn_w (head x k) reshaped to map",
            )
            self._save_overlay_grid_png(
                os.path.join(sample_dir, "attn_w_grid_overlay.png"),
                img_rgb,
                attn_tiles,
                titles=attn_titles,
                suptitle="attn_w overlay (head x k)",
                alpha=0.45,
            )

            # --- 2) enc_map を K タイルで1枚 ---
            enc_tiles = enc_map_cpu[b].numpy()  # (K,Hp,Wp)
            enc_titles = [f"k{k}" for k in range(K)]
            self._save_heatmap_grid_png(
                os.path.join(sample_dir, "enc_map_grid.png"),
                enc_tiles,
                titles=enc_titles,
                suptitle="enc_map (K channels)",
            )
            self._save_overlay_grid_png(
                os.path.join(sample_dir, "enc_map_grid_overlay.png"),
                img_rgb,
                enc_tiles,
                titles=enc_titles,
                suptitle="enc_map overlay",
                alpha=0.45,
            )

            # --- 3) pred_attn_map を K タイルで1枚 ---
            pred_tiles = pred_map_cpu[b].numpy()  # (K,Hp,Wp)
            pred_titles = [f"k{k}" for k in range(K)]
            self._save_heatmap_grid_png(
                os.path.join(sample_dir, "pred_attn_map_grid.png"),
                pred_tiles,
                titles=pred_titles,
                suptitle="pred_attn_map (K channels)",
            )
            self._save_overlay_grid_png(
                os.path.join(sample_dir, "pred_attn_map_grid_overlay.png"),
                img_rgb,
                pred_tiles,
                titles=pred_titles,
                suptitle="pred_attn_map overlay",
                alpha=0.45,
            )

        self.vis_cnt += 1

    def _grid_dims(self, n: int):
        """n枚をなるべく正方形に並べる (rows, cols)"""
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        return rows, cols

    def _save_heatmap_grid_png(
        self, path: str, maps: np.ndarray, titles=None, suptitle: str = "", vmin=None, vmax=None
    ):
        """
        maps: (N,H,W)
        titles: list[str] or None
        """
        assert maps.ndim == 3, f"maps must be (N,H,W), got {maps.shape}"
        N, H, W = maps.shape
        rows, cols = self._grid_dims(N)

        if vmin is None:
            vmin = float(np.min(maps))
        if vmax is None:
            vmax = float(np.max(maps))

        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
        axes = np.array(axes).reshape(rows, cols)

        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.axis("off")
            if i >= N:
                continue
            ax.imshow(maps[i], vmin=vmin, vmax=vmax)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=9)

        if suptitle:
            fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _save_overlay_grid_png(
        self,
        path: str,
        img_rgb: np.ndarray,
        maps: np.ndarray,
        titles=None,
        suptitle: str = "",
        alpha=0.45,
        vmin=None,
        vmax=None,
    ):
        """
        img_rgb: (H,W,3) uint8
        maps: (N,H,W)
        """
        assert maps.ndim == 3, f"maps must be (N,H,W), got {maps.shape}"
        N, H, W = maps.shape
        rows, cols = self._grid_dims(N)

        if vmin is None:
            vmin = float(np.min(maps))
        if vmax is None:
            vmax = float(np.max(maps))

        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
        axes = np.array(axes).reshape(rows, cols)

        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.axis("off")
            if i >= N:
                continue
            ax.imshow(img_rgb)
            ax.imshow(maps[i], alpha=alpha, vmin=vmin, vmax=vmax)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=9)

        if suptitle:
            fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
