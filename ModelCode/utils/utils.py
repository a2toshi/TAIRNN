import importlib
import os
import random

import numpy as np
import torch


def check_path(path, mkdir=False):
    """
    checks given path is existing or not
    """
    if path[-1] == "/":
        path = path[:-1]

    if not os.path.exists(path):
        if mkdir:
            os.mkdir(path)
        else:
            raise ValueError("%s does not exist" % path)
    return path


def set_logdir(log_dir, tag):
    return check_path(os.path.join(log_dir, tag), mkdir=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 以下は再現性を完全に保つため
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:  # スケジューラの状態を読み込む
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_loss = checkpoint.get("train_loss", None)
        test_loss = checkpoint.get("test_loss", None)
        print(f"Resuming from epoch {start_epoch}, train_loss={train_loss}, test_loss={test_loss}")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0


def load_model(model_name: str, class_name: str):
    """
    model_name: .py を除いたモジュール名
    class_name: モジュール内のクラス名
    """
    module = importlib.import_module(f"ModelCode.model.{model_name}")

    return getattr(module, class_name)
