import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
from dynaconf import Dynaconf
from sklearn.preprocessing import StandardScaler

import logging
from datetime import datetime

from arguments import _parse_args
from setting import configurate, torch_setting
from function import data_task_domain, get_dataloader, test, measure_variance_collapse
from data_ import get_domain_data, get_domain_acc_data
from train import run_batch_BCE as run_batch

# --- FIX 1: Add compute_layerwise_lambdas to imports ---
from models import Classifier, interpolate_models, loss_barrier_between_two_finals, compute_layerwise_lambdas
from metrics_cl import compute_cl_metrics

import logging
from datetime import datetime


def setup_logger(alpha, method, seed, log_dir, also_console):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alpha_str = str(alpha).replace("/", "_").replace(" ", "")
    method_str = str(method).replace("/", "_").replace(" ", "")
    log_path = os.path.join(log_dir, f"alpha{alpha_str}_{method_str}_seed{seed}_{ts}.txt")

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

    logger, log_path = setup_logger(args.alpha, args.method, args.seed_, log_dir="./logs", also_console=True)

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
        X_train_t, train_loader, X_test_t, test_loader, scaler = \
            data_task_domain(config, scaler, month_name)
        
        C.train()
        
        # 2. Train
        for epoch in tqdm(range(config.epochs)):
            for inputs, labels in train_loader:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                run_batch(config, C_optimizer, C, criterion, inputs, labels)

        # ====================================================
        #           DIAGNOSIS & INTERPOLATION BLOCK
        # ====================================================
        current_Classifier = deepcopy(C)
        
        if task_id > 0:
            logger.info(f"\n--- Running Diagnostics (Task {task_id-1} -> {task_id}) ---")
            
            # A. Create Interpolated Model (Midpoint)
            C_interp = interpolate_models(past_Classifier, current_Classifier, alpha=args.alpha, method=args.method)
            C_interp.to(device)

            # B. Measure Loss Barrier
            # --- FIX 2: Correct arguments and return unpacking ---
            curve, barrier, _ = loss_barrier_between_two_finals(
                model_prev_final=past_Classifier,        # Correct name
                model_curr_final=current_Classifier,     # Correct name
                dataloader=test_loader,
                criterion=criterion,
                device=device,
                compute_layerwise_lambdas_fn=compute_layerwise_lambdas # Passed explicitly
            )

            logger.info(f"   [Barrier] Max Spike: {barrier:.5f}")

            # Save Barrier Curve
            curve_data = {
                'alpha': [p['s'] for p in curve], # Note: 's' is the key in your definition, not 'alpha'
                'loss': [p['loss'] for p in curve],
                'acc': [p['acc'] for p in curve]
            }
            np.savez(f"diagnostics_data/barrier_{task_id-1}_to_{task_id}.npz", **curve_data)

            # C. Measure Variance Collapse
            collapse_ratio = measure_variance_collapse(
                prev_model=past_Classifier,
                curr_model=current_Classifier,
                interp_model=C_interp,
                dataloader=train_loader,
                device=device,
                layer_name='block2' 
            )
            
            # D. (Optional) Apply Interpolation to Main Model
            # C = C_interp 
            # C.to(device)

        # Update Past Model
        past_Classifier = deepcopy(C)

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

        R_prefix = R[:task_id+1, :task_id+1]
        m = compute_cl_metrics(R_prefix)
        logger.info(f"   [Metrics] ACC: {m['ACC']*100:.2f}% | BWT: {m['BWT']*100:.2f}%")
        logger.info(f"   [Metrics] ACC_CUMUL (actual): {acc_cumul}")

    # Final Save
    logger.info(f"The Accuracy for each task: {acc_cumul}")
    logger.info(f"The Global Average: {sum(acc_per_task) / len(acc_per_task)}")
    np.save(f"R_matrix_domain_months_{args.alpha}_{args.method}_{args.seed_}.npy", R)
    logger.info("\nDone.")


if __name__ == "__main__":
    main()