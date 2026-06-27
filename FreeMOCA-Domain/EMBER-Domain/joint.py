import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
from dynaconf import Dynaconf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import logging
from datetime import datetime

from arguments import _parse_args
from setting import configurate, torch_setting
from function import data_task_domain, get_dataloader, test, measure_variance_collapse, data_task_domain_joint
from data_ import get_domain_data, get_domain_acc_data
from train import run_batch_BCE as run_batch

# --- FIX 1: Add compute_layerwise_lambdas to imports ---
from models import Classifier, interpolate_models, loss_barrier_between_two_finals, compute_layerwise_lambdas
from metrics_cl import compute_cl_metrics


def setup_logger(seed, log_dir, also_console):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"Joint_seed{seed}_{ts}.txt")

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if also_console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.info(f"Logging to: {log_path}")
    return logger, log_path


def main():
    # --- Setup ---
    config = Dynaconf()
    args = _parse_args()
    configurate(args, config)
    torch_setting(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger, log_path = setup_logger(args.seed_, log_dir="./logs", also_console=True)

    # --- Model ---
    C = Classifier()
    C.to(device)

    C_optimizer = optim.SGD(C.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = StandardScaler()

    tasks = config.domains 
    config.nb_task = len(tasks)
    R = np.full((config.nb_task, config.nb_task), np.nan)

    # Create folder for plots/data
    os.makedirs("diagnostics_data", exist_ok=True)

    acc_per_task = []

    # ================================
    #        MAIN TRAINING LOOP
    # ================================
    for task_id, month_name in enumerate(tasks):
        logger.info(f"\n{'='*40}")
        logger.info(f" TASK {task_id} : {month_name}")
        logger.info(f"{'='*40}")
        config.task = task_id

        # 1. Load Data
        X_train_t, train_loader, scaler = \
            data_task_domain_joint(config, scaler, month_name, tasks, task_id)
        
        C.train()
        
        # 2. Train
        for epoch in tqdm(range(config.epochs)):
            for inputs, labels in train_loader:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                run_batch(config, C_optimizer, C, criterion, inputs, labels)

        # ====================================================
        #               EVALUATION LOOP
        # ====================================================
        logger.info(f"\n--- Evaluation after Task {task_id} ---")
        C.eval()

        for j, target_month in enumerate(tasks):
            X_test_j, Y_test_j = get_domain_data(config.data_root, target_month, is_train=False)
            test_loader_j, _ = get_dataloader(
                X_test_j, Y_test_j,
                batchsize=config.batchsize,
                n_class=config.init_classes,
                scaler=scaler,
                train=False
            )

            with torch.no_grad():
                acc_pct = test(config, C, test_loader_j)
                
            R[task_id, j] = acc_pct / 100.0
            
            if j <= task_id + 1:
                logger.info(f"   Test on {target_month}: {acc_pct:.2f}%")

        # Metrics
        X_test_acc_cumul, Y_test_acc_cumul = get_domain_acc_data(config.data_root, tasks, task_id)
        test_acc_cumul_loader, _ = get_dataloader(
                X_test_acc_cumul, Y_test_acc_cumul,
                batchsize=config.batchsize,
                n_class=config.init_classes,
                scaler=scaler,
                train=False
            )
        with torch.no_grad():
            acc_cumul = test(config, C, test_acc_cumul_loader)
        acc_per_task.append(acc_cumul)

        # Metrics
        R_prefix = R[:task_id+1, :task_id+1]
        m = compute_cl_metrics(R_prefix)
        logger.info(f"   [Metrics] ACC: {m['ACC']*100:.2f}% | BWT: {m['BWT']*100:.2f}%")
        logger.info(f"   [Metrics] ACC_CUMUL (actual): {acc_cumul}")

    # Final Save
    np.save("R_matrix_domain_months.npy", R)
    logger.info(f"The Global Average: {sum(acc_per_task) / len(acc_per_task)}")
    logger.info("\nDone.")


if __name__ == "__main__":
    main()