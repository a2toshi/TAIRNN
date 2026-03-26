import torch
import torch.nn as nn
from ModelCode.utils import LossScheduler, tensor2numpy


def repulsion_loss(points, margin=0.08):  # points: (B,K,2), 0..1正規化前提
    B, K, _ = points.shape
    D = torch.cdist(points, points, p=2) + torch.eye(K, device=points.device)[None] * 1e6
    # i<j のみ
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


class fullBPTTtrainer:

    def __init__(self, model, optimizer, loss_weights=[1.0, 1.0], joint_dim=8, device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=300, curve_name="s")
        self.model = model.to(self.device)
        self.joint_dim = joint_dim

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

    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        # 以下3行追加
        total_img_loss = 0.0
        total_joint_loss = 0.0
        total_pt_loss = 0.0

        total_pos_loss = 0.0
        total_quat_loss = 0.0
        total_grip_loss = 0.0
        total_rp_loss = 0.0

        for n_batch, ((x_img, x_joint), (y_img, y_joint)) in enumerate(data):
            if "cpu" in self.device:
                x_img = x_img.to(self.device)
                y_img = y_img.to(self.device)
                x_joint = x_joint.to(self.device)
                y_joint = y_joint.to(self.device)

            state = None
            yi_list, yv_list = [], []
            dec_pts_list, enc_pts_list = [], []
            self.optimizer.zero_grad(set_to_none=True)

            rp_steps = []

            # シーケンスデータの各タイムステップごとの計算
            for t in range(x_img.shape[1] - 1):
                _yi_hat, _yv_hat, enc_ij, dec_ij, state = self.model(x_img[:, t], x_joint[:, t], state)
                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                enc_pts_list.append(enc_ij)
                dec_pts_list.append(dec_ij)

                rp_steps.append(peak_weighted_cosine_diversity(self.model.attn_maps, top_p=0.2, margin=0.3))

                del _yi_hat, _yv_hat, enc_ij, dec_ij
                torch.cuda.empty_cache()

            # 予測結果をスタック
            yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
            yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))

            # 損失計算
            img_loss = nn.MSELoss()(yi_hat, y_img[:, 1:]) * self.loss_weights[0]
            y_joint_sliced = y_joint[:, :, : self.joint_dim]  # カメラ座標系座標を取り除く
            y_joint_target = y_joint_sliced[:, 1:]
            pos_loss_val = nn.MSELoss()(yv_hat[:, :, :3], y_joint_target[:, :, :3])
            quat_loss_val = nn.MSELoss()(yv_hat[:, :, 3:7], y_joint_target[:, :, 3:7])
            grip_loss_val = nn.MSELoss()(yv_hat[:, :, 7:], y_joint_target[:, :, 7:])
            joint_loss = (pos_loss_val + quat_loss_val + grip_loss_val) * self.loss_weights[1]
            pt_loss = nn.MSELoss()(torch.stack(dec_pts_list[:-1]), torch.stack(enc_pts_list[1:])) * self.scheduler(
                self.loss_weights[2]
            )
            rp_raw = torch.stack(rp_steps).mean() if rp_steps else torch.zeros((), device=x_img.device)
            rp_loss = rp_raw * self.scheduler(self.loss_weights[3])

            loss = img_loss + joint_loss + pt_loss + rp_loss

            # --- 記録用加算 ---
            total_loss += tensor2numpy(loss)
            total_img_loss += tensor2numpy(img_loss)
            total_joint_loss += tensor2numpy(joint_loss)
            total_pt_loss += tensor2numpy(pt_loss)
            total_pos_loss += tensor2numpy(pos_loss_val)
            total_quat_loss += tensor2numpy(quat_loss_val)
            total_grip_loss += tensor2numpy(grip_loss_val)
            total_rp_loss += tensor2numpy(rp_loss)

            # 勾配計算とパラメータ更新
            if training:
                loss.backward()
                self.optimizer.step()

        num_batches = n_batch + 1
        return (
            total_loss / num_batches,
            total_img_loss / num_batches,
            total_joint_loss / num_batches,
            total_pt_loss / num_batches,
            total_rp_loss / num_batches,
            total_pos_loss / num_batches,
            total_quat_loss / num_batches,
            total_pt_loss / num_batches,
        )
