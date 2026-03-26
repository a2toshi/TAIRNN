import torch
import torch.nn as nn
from ModelCode.utils import LossScheduler, tensor2numpy


class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

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

            # シーケンスデータの各タイムステップごとの計算
            for t in range(x_img.shape[1] - 1):
                _yi_hat, _yv_hat, enc_ij, dec_ij, state = self.model(x_img[:, t], x_joint[:, t], state)
                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                enc_pts_list.append(enc_ij)
                dec_pts_list.append(dec_ij)

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
            loss = img_loss + joint_loss + pt_loss

            # --- 記録用加算 ---
            total_loss += tensor2numpy(loss)
            total_img_loss += tensor2numpy(img_loss)
            total_joint_loss += tensor2numpy(joint_loss)
            total_pt_loss += tensor2numpy(pt_loss)
            total_pos_loss += tensor2numpy(pos_loss_val)
            total_quat_loss += tensor2numpy(quat_loss_val)
            total_grip_loss += tensor2numpy(grip_loss_val)

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
            total_pos_loss / num_batches,
            total_quat_loss / num_batches,
            total_pt_loss / num_batches,
        )
