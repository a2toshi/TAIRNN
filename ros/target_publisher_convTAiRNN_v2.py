#!/usr/bin/env python3

# export PYTHONPATH=$PYTHONPATH:/home/kklab003/h-taira/udp_send_test/kklab_auto_ws_v2/src/ros_send_test_pkg/script

import argparse
import os
import socket
import sys
import time
from datetime import datetime
from threading import Lock, Thread

import matplotlib
matplotlib.use("Agg")  # ← Tk を使わない
import matplotlib.pyplot as plt

import cv2
import numpy as np
import rospy
import torch
from std_msgs.msg import String

from eipl.model import TAiRNN, TAiRNNv2, TAiRNNv3, TAiRNNv5, TAiRNNv8, ConvTAiRNNv2, ConvTAiRNNv3, ConvTAiRNNv4, ConvTAiRNNv8
from eipl.utils import restore_args

latest_image = None
latest_position = None

ideal_position = [0.904348,-0.055210,-0.422838,0.017747]
# ideal_position = [0.798291,-0.010318,-0.602076,0.0114086]
# 0.906738,-0.052352,-0.418194,0.014083
# ideal_position = [0, 1, 0, 0]
ideal_position = np.array(ideal_position, dtype=np.float32)
cnt = 0
data_min_v2 = [600.935486,-88.666893,-507.579956,0.747684,-0.332047,-0.642025,-0.157874,0.088849,1349.69434,-93.915103,-202.438447]
data_max_v2 = [674.543884,-32.2561,-439.907532,0.91828,0.089902,-0.381841,0.165229,1.0,1448.335555,30.731925,-21.644632]
data_min = [600.935486,-88.666893,-507.579956,0.088849,1349.69434,-93.915103,-202.438447]
data_max = [674.543884,-32.2561,-439.907532,1.0,1448.335555,30.731925,-21.644632]

# data_min_v2 = [578.743164,-84.545258,-497.299347,0.747684,-0.332047,-0.642025,-0.157874,0.088849,1349.029874,-190.676842,-350.08075]
# data_max_v2 = [666.928406,-13.106107,-457.483795,0.91828,0.089902,-0.381841,0.165229,1.0,1422.656591,-75.16982,-131.612438]
# data_min = [578.743164,-84.545258,-497.299347,0.088849,1349.029874,-190.676842,-350.08075]
# data_max = [666.928406,-13.106107,-457.483795,1.0,1422.656591,-75.16982,-131.612438]

data_min = np.array(data_min, dtype=np.float32)
data_max = np.array(data_max, dtype=np.float32)
data_min_v2 = np.array(data_min_v2, dtype=np.float32)
data_max_v2 = np.array(data_max_v2, dtype=np.float32)

img_size = 128
joint_dim = 4

image_lock = Lock()


def save_attention_map_image(attn_map, k_dim, Hp, Wp, save_path):
    """Attention Mapを画像として保存する関数"""
    attn_map_reshaped = attn_map.reshape(k_dim, Hp, Wp)

    num_cols = k_dim
    num_rows = 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    if num_cols == 1:
        axes = [axes]

    for i in range(k_dim):
        ax = axes[i]
        im = ax.imshow(attn_map_reshaped[i], cmap="jet")
        ax.set_title(f"Attn k={i}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)



def send_udp_message(message, ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode(), (ip, port))
        rospy.loginfo(f"Sent message: {message} to {ip}:{port}")
    except Exception as e:
        rospy.logerr(f"Error sending message: {e}")
    finally:
        sock.close()


def tensor2numpy(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()


def preprocess_image(image):
    # TODO:Pillowに変更?
    height, width = image.shape[:2]
    crop_width, crop_height = 1240, 1000
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2   

    cropped_image = image[start_y : start_y + crop_height, start_x : start_x + crop_width]
    resized_image = cv2.resize(cropped_image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return resized_image


def webcam_capture():
    global latest_image, cnt, captured_img_dir
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        rospy.logerr("Could not open webcam.")
        return

    # 空読み
    warmup_frames = 5
    rospy.loginfo(f"Warming up camera for {warmup_frames} frames...")
    for _ in range(warmup_frames):
        cap.read()
    rospy.loginfo("Camera warmup complete.")

    while not rospy.is_shutdown():
        cnt += 1
        ret, frame = cap.read()
        # cv2.imwrite(captured_img_dir + f"/debug_image_{cnt}.png", frame)
        if ret:
            # frame = cv2.flip(frame, 0) #TODO:上下反転
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            processed_image = preprocess_image(frame)
            with image_lock:
                latest_image = processed_image


    cap.release()


def callback(data):
    global latest_position
    # rospy.loginfo("Received data")
    message = data.data.split(",")
    # 実データx,y,z　＋　モデルが出力したidealなクォータニオン + カメラ座標系での位置座標
    position = [float(x) for x in message[:8]]

    # position = position
    position = position + [float(message[i]) for i in range(9, 12)]

    position = np.array(position, dtype=np.float32)
    latest_position = (position - data_min_v2) / (data_max_v2 - data_min_v2)


def convert_image_to_tensor():
    global latest_image
    with image_lock:
        image_array = latest_image
    if image_array is not None:
        return torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    else:
        return None


def target_publisher(position: str):
    # rospy.init_node('udp_sender_node', anonymous=True)

    str1_mono = position + ",0"  # TODO:message形式変更に伴い調整
    str2_mono = "-100,30,0,0,1,0,0,1"
    str_default = "0,0,0,0,1,0,0,0,0"

    str1 = str1_mono + "," + str_default + "," + str_default + "\n"
    str2 = str2_mono + "," + str_default + "," + str_default + "\n"

    ip = rospy.get_param("~ip", "127.0.0.1")
    port = rospy.get_param("~port", 30002)

    rospy.loginfo(f"Starting UDP sender with ip: {ip}, port: {port}")

    try:
        # str1を送信
        send_udp_message(str1, ip, port)
        # send_udp_message(str2, ip, port)
        # rospy.sleep(15)  # 5秒間スリープ

        # # str2を送信
        # send_udp_message(str2, ip, port)
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down UDP sender node.")
    time.sleep(0.6)


if __name__ == "__main__":
    filename = "src/ros_send_test_pkg/script/log/20251231_1500_00/compatible_model_2000.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, default=filename)
    parser.add_argument("--input_param", type=float, default=1.0)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = "debug_image/TAiRNN/" + timestamp
    os.makedirs(folder_path, exist_ok=True)
    captured_img_dir = os.path.join(folder_path, "captured_images")
    predicted_img_dir = os.path.join(folder_path, "predicted_images")
    os.makedirs(captured_img_dir, exist_ok=True)
    os.makedirs(predicted_img_dir, exist_ok=True)

    # publish_positionを保存するCSVファイルのパスを定義
    position_log_path = os.path.join(folder_path, "published_positions.csv")

    # ファイルを開き、ヘッダーを書き込む
    # このファイルオブジェクトをループの外で定義し、ループ内で使用する
    log_file = open(position_log_path, "w")
    log_file.write("pos_x,pos_y,pos_z\n")

    # restore parameters
    dir_name = os.path.split(args.model_pth)[0]
    params = restore_args(os.path.join(dir_name, "args.json"))

    # define model
    model = ConvTAiRNNv8(
        rec_dim=params["rec_dim"],
        joint_dim=joint_dim,
        k_dim=params["k_dim"],
        heatmap_size=params["heatmap_size"],
        temperature=params["temperature"],
        im_size=[img_size, img_size],
    )

    # load weight
    ckpt = torch.load(args.model_pth, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rospy.init_node("robot_listener", anonymous=True)
    rospy.Subscriber("robot_data", String, callback)
    # print(latest_position)

    cam_thread = Thread(target=webcam_capture)
    cam_thread.start()
    # rospy.spin()

    # Inference
    # Set the inference frequency; for a 10-Hz in ROS system, set as follows.
    freq = 0.2  # 10Hz
    rate = rospy.Rate(freq)
    # image_list, joint_list = [], []
    state = None
    nloop = 200  # freq * 20 sec

    try:
        while latest_image is None:
            time.sleep(1)
        
        h_norm_log = []
        enc_pts_log = []
        dec_pts_log = []
        attn_map_dir = os.path.join(folder_path, "attn_maps_sequence")
        os.makedirs(attn_map_dir, exist_ok=True)

        for loop_ct in range(nloop):
            if rospy.is_shutdown():
                break
            start_time = time.time()

            if latest_image is None:
                print("image hoge")
            if latest_position is None:
                print("position hoge")

            if latest_image is not None and latest_position is not None:
                print("latest_position:")
                print(latest_position)
                image = convert_image_to_tensor()
                overlay_img = image[0].permute(1, 2, 0).cpu().numpy()
                overlay_img = (overlay_img * 255).astype(np.uint8)
                overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)

                pos_wo_quat = np.concatenate([latest_position[:3], latest_position[7:11]])  #
                print("pos_wo_quat:")
                print(pos_wo_quat)
                position = torch.tensor(pos_wo_quat).unsqueeze(0)

                y_image, y_position, enc_map, pred_map, state = model(image, position, state)

                if loop_ct % 1 == 0:
                    # キャプチャーした入力画像を保存
                    # #    `image`は(1, C, H, W)のTensorなので、整形してNumpy配列に戻す
                    captured_img_np = image[0].permute(1, 2, 0).cpu().numpy()
                    captured_img_to_save = (captured_img_np * 255).astype(np.uint8)
                    captured_img_to_save = cv2.cvtColor(captured_img_to_save, cv2.COLOR_RGB2BGR)
                    captured_path = os.path.join(captured_img_dir, f"captured_{loop_ct:04d}.png")
                    cv2.imwrite(captured_path, captured_img_to_save)

                    # 予測画像を保存
                    #    `y_image_tensor`も同様に整形して保存
                    pred_img_np = y_image[0].permute(1, 2, 0).detach().cpu().numpy()
                    pred_img_to_save = (pred_img_np * 255).astype(np.uint8)
                    pred_img_to_save = cv2.cvtColor(pred_img_to_save, cv2.COLOR_RGB2BGR)
                    predicted_path = os.path.join(predicted_img_dir, f"predicted_{loop_ct:04d}.png")
                    cv2.imwrite(predicted_path, pred_img_to_save)



                h_norm_log.append(torch.linalg.norm(state[0].detach()).item())

                save_attention_map_image(
                    attn_map=enc_map[0].squeeze(0).detach().cpu().numpy(),
                    k_dim=params["k_dim"],
                    Hp=model.Hp,
                    Wp=model.Wp,
                    save_path=os.path.join(attn_map_dir, f"enc_map_{loop_ct:04d}.png"),
                )
                save_attention_map_image(
                    attn_map=pred_map[0].squeeze(0).detach().cpu().numpy(),
                    k_dim=params["k_dim"],
                    Hp=model.Hp,
                    Wp=model.Wp,
                    save_path=os.path.join(attn_map_dir, f"pred_map_{loop_ct:04d}.png"),
                )

                pred_position = tensor2numpy(y_position[0][:joint_dim])
                pred_position[:3] += latest_position[:3]
                print(pred_position,"\n")

                # ideal_position = pred_position[3:7].tolist()
                pred_position = pred_position * (data_max[:4] - data_min[:4]) + data_min[:4]

                pub_position = np.concatenate([pred_position[:3], ideal_position, pred_position[-1:]], axis=0)
                print(pub_position)
                # pub_position = pub_position * (data_max_v2[:8] - data_min_v2[:8]) + data_min_v2[:8]



                publish_position = ",".join(map(str, pub_position))
                # log_position = pred_position[:3] + latest_position
                log_position = ",".join(map(str, latest_position))


                # 計算した座標をCSVファイルに追記
                log_file.write(log_position + "\n")
                log_file.write(publish_position + "\n")
                # log_file.write(latest_position + "\n\n")

                target_publisher(publish_position)
                rate.sleep()

    finally:
        # ループが終了（または中断）した際にファイルを閉じる
        if log_file:
            log_file.close()
            rospy.loginfo(f"Position log saved to: {position_log_path}")

    # ループ終了後に結果をプロットして保存
    rospy.loginfo("Control loop finished. Generating analysis plots...")

    # NumPy配列に変換
    h_norm_log = np.array(h_norm_log)
    enc_pts_log = np.array(enc_pts_log)
    dec_pts_log = np.array(dec_pts_log)

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # 1. h のノルムをプロット
    axs[0].plot(h_norm_log)
    axs[0].set_title("L2 Norm of LSTM Hidden State (h) over time")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("L2 Norm")

    # 2. enc_pts をプロット (x座標とy座標を点ごとに色分け)
    for i in range(params["k_dim"]):
        axs[1].plot(enc_pts_log[:, i * 2], label=f"enc_pt_{i}_x")
        axs[1].plot(enc_pts_log[:, i * 2 + 1], label=f"enc_pt_{i}_y", linestyle="--")
    axs[1].set_title("Encoded Attention Points (enc_pts) over time")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Coordinate Value")
    axs[1].legend(ncol=params["k_dim"])

    # 3. dec_pts をプロット
    for i in range(params["k_dim"]):
        axs[2].plot(dec_pts_log[:, i * 2], label=f"dec_pt_{i}_x")
        axs[2].plot(dec_pts_log[:, i * 2 + 1], label=f"dec_pt_{i}_y", linestyle="--")
    axs[2].set_title("Decoded Attention Points (dec_pts) over time")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Coordinate Value")
    axs[2].legend(ncol=params["k_dim"])

    fig.tight_layout()
    analysis_plot_path = os.path.join(folder_path, "state_analysis_plot.png")
    plt.savefig(analysis_plot_path)
    plt.close(fig)

    rospy.loginfo(f"Analysis plot saved to: {analysis_plot_path}")
    rospy.loginfo(f"Attention map sequence saved in: {attn_map_dir}")