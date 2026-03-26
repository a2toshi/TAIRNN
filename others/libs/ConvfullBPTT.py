import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelCode.utils import LossScheduler, tensor2numpy


class ConvfullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, model, optimizer, loss_weights=[1.0, 1.0, 0.1], device="cpu", joint_dim=3):
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
        total_attn_loss = 0.0

        for n_batch, ((x_img, x_joint), (y_img, y_joint)) in enumerate(data):
            if "cpu" in self.device:
                x_img = x_img.to(self.device)
                y_img = y_img.to(self.device)
                x_joint = x_joint.to(self.device)
                y_joint = y_joint.to(self.device)

            state = None
            yi_list, yv_list = [], []
            input_map_list, pred_map_list = [], []
            self.optimizer.zero_grad(set_to_none=True)

            # シーケンスデータの各タイムステップごとの計算
            for t in range(x_img.shape[1] - 1):
                _yi_hat, _yv_hat, _input_map, _pred_map, state = self.model(x_img[:, t], x_joint[:, t], state)
                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                pred_map_list.append(_pred_map)
                input_map_list.append(_input_map)

                del _yi_hat, _yv_hat, _input_map, _pred_map
                torch.cuda.empty_cache()

            # 予測結果をスタック
            yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
            yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))

            # 損失計算
            y_joint_sliced = y_joint[:, :, : self.joint_dim]
            img_loss = nn.MSELoss()(yi_hat, y_img[:, 1:]) * self.loss_weights[0]
            joint_loss = nn.MSELoss()(yv_hat, y_joint_sliced[:, 1:]) * self.loss_weights[1]
            pred_maps = torch.stack(pred_map_list[:-1])
            target_maps = torch.stack(input_map_list[1:])

            # =====通常のMSE LOSS=====
            # attn_loss = nn.MSELoss()(pred_maps, target_maps.detach()) * self.scheduler(self.loss_weights[2])

            # =====KLDiv採用=====
            # kl_loss = nn.KLDivLoss(reduction="mean")
            # eps = 1e-7
            # attn_loss = kl_loss(pred_maps, target_maps.detach()) * self.scheduler(self.loss_weights[2])

            # =====エントロピー最小化=====
            attn_main = nn.MSELoss()(pred_maps, target_maps.detach())

            # エントロピー最小化
            ENT_W = 0.03  # エントロピー(集中度)の弱正則化
            TV_W = 0.002  # 総変動(なめらかさ)

            p = torch.sigmoid(pred_maps)
            # エントロピー最小化（広がりすぎ防止）
            ent = -(
                p.clamp(1e-6, 1 - 1e-6) * torch.log(p.clamp(1e-6, 1 - 1e-6))
                + (1 - p) * torch.log((1 - p).clamp(1e-6, 1 - 1e-6))
            ).mean()
            # TV
            tv = (p[..., 1:, :] - p[..., :-1, :]).abs().mean() + (p[..., :, 1:] - p[..., :, :-1]).abs().mean()
            attn_loss = self.scheduler(self.loss_weights[2]) * (attn_main + ent * ENT_W + tv * TV_W)

            loss = img_loss + joint_loss + attn_loss
            total_loss += tensor2numpy(loss)
            # 以下3行追加
            total_img_loss += tensor2numpy(img_loss)
            total_joint_loss += tensor2numpy(joint_loss)
            total_attn_loss += tensor2numpy(attn_loss)

            # 勾配計算とパラメータ更新
            if training:
                loss.backward()
                self.optimizer.step()

        num_batches = n_batch + 1
        return (
            total_loss / num_batches,
            total_img_loss / num_batches,
            total_joint_loss / num_batches,
            total_attn_loss / num_batches,
        )
