import torch
import torch.nn as nn
from ModelCode.utils import LossScheduler, tensor2numpy

# 変更点：


class MultiStepfullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, model, optimizer, loss_weights=[1.0, 1.0], device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=1500, curve_name="s")
        self.model = model.to(self.device)
        print("[FullBPTT: New Arc on 0704]")

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

        # --- 追加: 複数ステップ予測のホライゾン長 ---
        prediction_horizon = 5  # 例: 5ステップ先まで予測

        for n_batch, ((x_img, x_joint), (y_img, y_joint)) in enumerate(data):
            # ... (デバイス移動など) ...

            self.optimizer.zero_grad(set_to_none=True)

            # このバッチで計算した全ての損失を合算する変数
            img_loss = 0
            joint_loss = 0
            pt_loss = 0

            # シーケンスデータの各タイムステップをループ
            # (シーケンスの最後まで予測すると正解データがなくなるため、少し手前まで)
            for t in range(x_img.shape[1] - prediction_horizon):

                # --- モデルの呼び出し方を変更 ---
                # t時点のデータから、t+1 ~ t+H までのHステップを予測
                y_image_list_pred, y_joint_list_pred, _, dec_pts_list_pred, _ = self.model(
                    xi=x_img[:, t],
                    xv=x_joint[:, t],
                    state=None,  # 各予測開始時点でstateはリセット
                    pred_len=prediction_horizon,
                    input_mode="pred",
                )

                # --- 損失計算の修正 ---
                # Hステップ分の予測と正解を比較し、損失を合計
                for h_step in range(prediction_horizon):
                    # 予測値
                    yi_hat = y_image_list_pred[h_step]
                    yv_hat = y_joint_list_pred[h_step]

                    # 正解値 (t+1+h_step)
                    yi_true = y_img[:, t + 1 + h_step]
                    yv_true = y_joint[:, t + 1 + h_step, : self.model.joint_dim]

                    # 損失を計算
                    img_loss = nn.MSELoss()(yi_hat, yi_true) * self.loss_weights[0]
                    joint_loss = nn.MSELoss()(yv_hat, yv_true) * self.loss_weights[1]

                dec_pts_t_plus_1 = dec_pts_list_pred[0]

                # t+1 の正解画像から、正解のenc_ptsを計算
                _, _, enc_pts_t_plus_1_true, _, _ = self.model(
                    xi=x_img[:, t + 1],
                    xv=x_joint[:, t + 1],
                    state=None,
                    pred_len=1,
                    input_mode="real",  # 'real'モードで1ステップだけ実行
                )

                # 正解のenc_ptsと、予測したdec_ptsで損失を計算
                pt_loss += nn.MSELoss()(dec_pts_t_plus_1, enc_pts_t_plus_1_true[0])

            img_loss /= prediction_horizon
            joint_loss /= prediction_horizon
            pt_loss = pt_loss * self.scheduler(self.loss_weights[2])
            loss = img_loss + joint_loss + pt_loss
            total_loss += tensor2numpy(loss)
            # 以下3行追加
            total_img_loss += tensor2numpy(img_loss)
            total_joint_loss += tensor2numpy(joint_loss)
            total_pt_loss += tensor2numpy(pt_loss)

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
        )
