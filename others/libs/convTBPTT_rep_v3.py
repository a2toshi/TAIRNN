import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelCode.utils import LossScheduler, tensor2numpy

# 変更：attention mapのlossをMSEからKLDivに変更


def repulsion_loss(points, margin=0.08):  # points: (B,K,2), 0..1正規化前提
    B, K, _ = points.shape
    D = torch.cdist(points, points, p=2) + torch.eye(K, device=points.device)[None] * 1e6
    triu = torch.triu(torch.ones(K, K, device=points.device), diagonal=1).bool()
    dij = D[:, triu]  # (B, K*(K-1)/2)
    return torch.relu(margin - dij).pow(2).mean()


def peak_mask_top_p(A, top_p=0.2):
    # A: (B, K, N) 0..1 くらいの正規化済み（sharp後推奨）
    B, K, N = A.shape
    # 各(K)mapで上位p%のしきい値
    thresh = torch.quantile(A, 1.0 - top_p, dim=-1, keepdim=True)  # (B,K,1)
    M = (A >= thresh).float()  # (B,K,N) 0/1
    return M


def peak_weighted_cosine_diversity(A, top_p=0.2, margin=0.3, eps=1e-8):
    """
    A: (B,K,N) ・・・ sharp化後の Attention Map（各行は確率でなくてもOK）
    戻り値: スカラー損失（大きいほど多様性不足）
    """
    # 1) NaN/Inf 安全化
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) 強い領域のみ残すマスク
    M = peak_mask_top_p(A, top_p=top_p)
    A_masked = A * M

    # 3) 正規化（重み付き L2 ノルム）
    #    ※完全にゼロの行は落ちないように eps を与える
    denom = torch.sqrt((A_masked**2).sum(dim=-1, keepdim=True) + eps)
    A_norm = A_masked / denom

    # 4) K×K のコサイン類似（バッチ毎）
    #    sim_ij = <a_i, a_j> on masked space
    sim = torch.einsum("bkn,bmn->bkm", A_norm, A_norm)  # (B,K,K)
    # 対角は無視、上三角だけ対象
    Kdim = A.shape[1]
    iu = torch.triu_indices(Kdim, Kdim, 1, device=A.device)
    sim_pairs = sim[:, iu[0], iu[1]]  # (B, K*(K-1)/2)

    # 5) margin ヒンジ（似すぎを罰する）
    loss = F.relu(sim_pairs - margin).pow(2).mean()
    return loss


def weighted_map_loss(pred_map, target_map, weight_peak=None):  # weight_peakは使わなくなる
    """
    分布間距離 (KL Divergence) を計算する
    pred_map: (B, K, H, W)  Sum to 1 (Softmax済み)
    target_map: (B, K, H, W) Sum to 1 (Softmax済み)
    """
    eps = 1e-8

    # KL(target || pred) = target * (log(target) - log(pred))
    # predが0になると無限大に発散するので eps を足す

    loss = torch.sum(target_map * (torch.log(target_map + eps) - torch.log(pred_map + eps)), dim=(2, 3))

    return loss.mean()


def get_temperature(epoch, max_epochs, start_temp=1.0, end_temp=1e-2):
    """
    エポックが進むにつれて温度を指数関数的に下げる関数
    Epoch 0: start_temp (1.0)
    Epoch Max: end_temp (0.0001)
    """
    if epoch >= max_epochs:
        return end_temp

    # 指数減衰の計算: T = T_start * (decay_rate ^ epoch)
    # decay_rate = (T_end / T_start) ^ (1 / max_epochs)

    decay_rate = (end_temp / start_temp) ** (1 / max_epochs)
    current_temp = start_temp * (decay_rate**epoch)

    return current_temp


# ★追加: S字カーブ生成関数
def s_curve_interpolation(self, start, end, num_points):
    t = np.linspace(0, 1, num_points)
    # S字の変化量
    x = start + (end - start) * (t - np.sin(2 * np.pi * t) / (2 * np.pi))
    return x


class TBPTT_rep_trainer:

    def __init__(self, model, optimizer, loss_weights, joint_dim=8, truncation_length=30, device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=100, curve_name="s")
        self.scheduler_pos = LossScheduler(decay_end=100, curve_name="s")
        self.scheduler_temp = LossScheduler(decay_end=100, curve_name="s")
        self.model = model.to(self.device)
        self.joint_dim = joint_dim
        self.truncation_length = truncation_length

        # 温度スケジュールの設定
        self.temp_start = 1.0  # 最初はぼんやり (勾配を通しやすくする)
        self.temp_end = 1e-2  # 最後はカッチリ (Hard Attentionに近づける)
        # 全エポックの80%くらいの時点で最小値に到達するようにすると良い

    def save(self, epoch, loss, savename, scheduler_state_dict=None):
        save_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": loss[0],
            "test_loss": loss[1],
        }
        if scheduler_state_dict:  # スケジューラの状態があれば追加
            save_dict["scheduler_state_dict"] = scheduler_state_dict
        torch.save(save_dict, savename)

    def process_epoch(self, data, epoch, num_epochs, data_max, data_min, training=True):
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        total_img_loss = 0.0
        total_joint_loss = 0.0
        total_pt_loss = 0.0
        total_rp_loss = 0.0

        # 追加: 成分別 joint loss
        total_pos_loss = 0.0  # Δpos(0:3)
        total_quat_loss = 0.0  # quat(3:7)
        total_grip_loss = 0.0  # g(7:8)

        # ★追加: XYZ個別のLoss監視用
        total_pos_x_loss = 0.0
        total_pos_y_loss = 0.0
        total_pos_z_loss = 0.0

        # mask確率
        p_seq = 0
        # p_seq = min(0.5, (epoch + 1) / 500)

        current_temp = self.temp_start + (self.temp_end - self.temp_start) * self.scheduler_temp(1.0)
        self.model.temp_sche = current_temp

        for n_batch, ((x_img, x_joint), (y_img, y_joint)) in enumerate(data):
            state = None
            self.optimizer.zero_grad(set_to_none=True)

            truncation_length = self.truncation_length
            total_steps = x_img.shape[1] - 1
            chunk_losses = 0.0

            yi_list, yv_list = [], []
            enc_maps_list, pred_maps_list = [], []
            rp_steps = []

            batch_size = x_joint.shape[0]

            isMasked = torch.rand(batch_size, 1, device=self.device) < p_seq

            for t in range(total_steps):
                current_x_joint = x_joint[:, t]
                mask_token = torch.full_like(current_x_joint, 0.0)
                # マスクしたくないquat部分だけ元の値を残すマスクを作る
                mask_keep = torch.ones_like(current_x_joint, dtype=torch.bool)
                mask_keep[:, 0:3] = False  # pos
                mask_keep[:, 3:4] = True  # g
                masked_x_joint = torch.where(isMasked & ~mask_keep, mask_token, current_x_joint)

                _yi_hat, _yv_hat, enc_ij, dec_ij, state = self.model(x_img[:, t], masked_x_joint, state)
                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                enc_maps_list.append(enc_ij)
                pred_maps_list.append(dec_ij)
                rp_steps.append(peak_weighted_cosine_diversity(self.model.attn_maps, top_p=0.2, margin=0.3))

                is_last_step = t == total_steps - 1
                is_chunk_end = (t + 1) % truncation_length == 0

                if is_chunk_end or is_last_step:
                    # 予測結果をスタック
                    yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
                    yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))
                    chunk_steps = len(yi_list)

                    # 正解ラベルをスライス
                    start_t_in_chunk = t - (len(yi_list) - 1)

                    # 正解ラベル（y_img, y_joint）のスライス
                    y_img_chunk = y_img[:, start_t_in_chunk + 1 : t + 2]
                    y_joint_chunk = y_joint[:, start_t_in_chunk + 1 : t + 2]
                    y_joint_chunk_old = y_joint[:, start_t_in_chunk : t + 1]

                    # 損失計算
                    delta_pos = y_joint_chunk[:, :, :3] - y_joint_chunk_old[:, :, :3]  # (B,T,3)

                    grip_gt = y_joint_chunk[:, :, 3:4]
                    # grip_gt = y_joint_chunk[:, :, 7:8]

                    # quat_hat = yv_hat[:, :, 3:7]
                    grip_hat = yv_hat[:, :, 3:4]

                    # 画像損失
                    img_loss = F.mse_loss(yi_hat, y_img_chunk) * self.loss_weights[0]

                    # joint 側：pos は mm、quat/grip はこれまで通り正規化空間で
                    pos_weights = torch.tensor([1.0, 1.0, 1.0], device=self.device).view(1, 1, 3)
                    pos_loss = (pos_weights * (yv_hat[:, :, :3] - delta_pos).pow(2)).mean()
                    grip_loss = F.mse_loss(grip_hat, grip_gt)

                    joint_loss = (pos_loss + grip_loss * 1e-2) * self.scheduler_pos(
                        self.loss_weights[1]
                    )  # 変更: jointにもスケジューリングを入れる
                    # joint_loss = (pos_loss + grip_loss * 1e-2) * self.loss_weights[1]

                    # ---- ここからログ用の成分別 loss ----
                    with torch.no_grad():
                        # ログは「素の MSE」で OK
                        pos_loss_chunk = F.mse_loss(yv_hat[:, :, :3], delta_pos)
                        grip_loss_chunk = F.mse_loss(grip_hat, grip_gt)

                        total_pos_loss += tensor2numpy(pos_loss_chunk) / max(1, chunk_steps)
                        total_grip_loss += tensor2numpy(grip_loss_chunk) / max(1, chunk_steps)

                        diff = yv_hat[:, :, :3] - delta_pos  # (B, T, 3)
                        l_x = (diff[:, :, 0] ** 2).mean()  # X軸
                        l_y = (diff[:, :, 1] ** 2).mean()  # Y軸
                        l_z = (diff[:, :, 2] ** 2).mean()  # Z軸
                        total_pos_x_loss += tensor2numpy(l_x) / max(1, chunk_steps)
                        total_pos_y_loss += tensor2numpy(l_y) / max(1, chunk_steps)
                        total_pos_z_loss += tensor2numpy(l_z) / max(1, chunk_steps)
                    # ---- ログ用ここまで ----

                    # Map loss
                    if len(pred_maps_list) > 1:
                        # (T-1, B, K, H, W) -> (B*(T-1), K, H, W) にして一括計算
                        preds_stack = torch.stack(pred_maps_list[:-1]).flatten(0, 1)
                        targets_stack = torch.stack(enc_maps_list[1:]).flatten(0, 1)

                        pt_loss = weighted_map_loss(preds_stack, targets_stack) * self.scheduler(self.loss_weights[2])
                    else:
                        pt_loss = torch.tensor(0.0, device=self.device)

                    rp_raw = torch.stack(rp_steps).mean() if rp_steps else torch.zeros((), device=x_img.device)
                    # rp_loss = rp_raw * (self.loss_weights[3])
                    rp_loss = rp_raw * self.scheduler(self.loss_weights[3])

                    # 損失を合計
                    loss = img_loss + joint_loss + pt_loss + rp_loss

                    # 損失を（このチャンクのステップ数で重み付けして）蓄積
                    chunk_losses += loss * len(yi_list)  # ステップ数で重み付け

                    if training:
                        loss.backward()  # このチャンクの分だけ逆伝播
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)  # 勾配をリセット

                    if state is not None:
                        h, c = state
                        state = (h.detach(), c.detach())

                    yi_list, yv_list = [], []
                    enc_maps_list, pred_maps_list = [], []
                    rp_steps = []
                    isMasked = torch.rand(batch_size, 1, device=self.device) < p_seq

                    total_img_loss += tensor2numpy(img_loss) / max(1, chunk_steps)
                    total_joint_loss += tensor2numpy(joint_loss) / max(1, chunk_steps)
                    total_pt_loss += tensor2numpy(pt_loss) / max(1, chunk_steps)
                    total_rp_loss += tensor2numpy(rp_loss) / max(1, chunk_steps)

                    del _yi_hat, _yv_hat, enc_ij, dec_ij, yi_hat, yv_hat, loss

            total_loss += tensor2numpy(chunk_losses) / total_steps

        num_batches = n_batch + 1
        return (
            total_loss / num_batches,
            total_img_loss / num_batches,
            total_joint_loss / num_batches,
            total_pt_loss / num_batches,
            total_rp_loss / num_batches,
            total_pos_loss / num_batches,
            total_grip_loss / num_batches,
            total_pos_x_loss / num_batches,  # 追加
            total_pos_y_loss / num_batches,  # 追加
            total_pos_z_loss / num_batches,  # 追加
        )
