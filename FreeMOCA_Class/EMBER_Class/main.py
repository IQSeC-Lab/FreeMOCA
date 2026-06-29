import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
import os
import pandas
from sklearn.preprocessing import StandardScaler
import joblib
from function import test, get_dataloader, class_pick_rand
from function import measure_variance_collapse # ensure you imported it
from torch.utils.data import TensorDataset
from dynaconf import Dynaconf
from arguments import _parse_args
from setting import configurate, torch_setting
from _data import dataset
from train import data_task,data_task_joint, report_result
from train import run_batch_BCE as run_batch
import subprocess
import random
from models import Classifier, interpolate_models
from models import average_two_models
from metrics_cl import compute_cl_metrics
from function import get_iter_test_dataset_task
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def main():
    config = Dynaconf()
    args = _parse_args()
    configurate(args, config)
    torch_setting(config)

    ##############
    # EMBER DATA #
    ##############

    X_train, Y_train, X_test, Y_test = dataset(config)

    # ############################################
    # # data random arrange #
    # #############################################

    Y_train, Y_test = class_pick_rand(config, Y_train, Y_test)

    ###############################
    # Models and Hyper Parameters #
    ###############################


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    criterion = nn.CrossEntropyLoss()


    ###############################
    # continual learning training #
    ###############################
    ls_a = []

    scaler = StandardScaler()

    ###############################
    C = Classifier(config.init_classes)

    C.to(device)
    C_optimizer = optim.SGD(C.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    scaler = StandardScaler()


    T = config.nb_task
    R = np.full((T, T), np.nan, dtype=float)  # store accuracies in [0,1]

    def n_class_for_task(task_id):
        # output size after "expanding up to task_id"
        if task_id == 0:
            return config.init_classes
        return config.init_classes + task_id * config.n_inc


    for task in range(config.nb_task):
        config.n_class = config.init_classes + task * config.n_inc
        config.task = task

        # Load data
        X_train_t, Y_train_t, train_loader, X_test_t, Y_test_t, test_loader, scaler = data_task(config, X_train, Y_train, X_test, Y_test, scaler)
        config.nb_batch = int(len(X_train_t) / config.batchsize)
        C.to(device)

        if task > 0:
            C = C.expand_output_layer(config.init_classes, config.n_inc, task)
            C.to(device)

        for epoch in tqdm(range(config.epochs)):
            C.train()
            epoch_loss = 0

            for n, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.float().to(config.device)
                labels = labels.float().to(config.device)

                # Run training batch
                run_batch(config, C_optimizer, C, criterion, inputs, labels)

        # Evaluate
        current_Classifier = deepcopy(C)
        
        if task > 0:
            # print(f"\nInterpolating task {task-1} and task {task}")
            # C = interpolate_models(past_Classifier, current_Classifier, alpha=0.5, method="linear")
            # C = interpolate_models(
            #     past_Classifier, current_Classifier,
            #     alpha = 0.1,
            #     method="linear",
            #     lambda_min=0.1,
            #     lambda_max=0.9
            # )
            # C.to(device)
            C = interpolate_models(
            past_Classifier,
            current_Classifier,
            alpha=None,          # or leave out → use only layerwise λ_l
            method="linear",
            lambda_min=0.3,
            lambda_max=0.7
        )
            # C = average_two_models(past_Classifier, current_Classifier)
            C.to(device)
            # --- INSERT DIAGNOSIS HERE ---
            # We pass train_loader (current task data) as the probe
            # Using 'block2' as it is the deepest convolutional layer in your models.py
            from models import compute_layerwise_lambdas, loss_barrier_between_two_finals

            curve2, barrier2, _ = loss_barrier_between_two_finals(
                model_prev_final=past_Classifier,  # this is tilde theta_{t-1}
                model_curr_final=C,                # this is tilde theta_t (after interpolation)
                dataloader=test_loader,            # use evaluation data for barrier
                criterion=criterion,
                device=device,
                compute_layerwise_lambdas_fn=compute_layerwise_lambdas,
                lambda_min=0.3,
                lambda_max=0.7,
                k=3.0,
                num_points=11
            )

            print(f"[Barrier final->final] task {task-1}->{task}: {barrier2:.6f}")

            os.makedirs("barrier_curves", exist_ok=True)

            s_vals = np.array([p["s"] for p in curve2], dtype=float)
            loss_vals = np.array([p["loss"] for p in curve2], dtype=float)
            acc_vals  = np.array([p["acc"] for p in curve2], dtype=float)

            np.savez(
                f"barrier_curves/curve_{task-1}_to_{task}.npz",
                s=s_vals,
                loss=loss_vals,
                acc=acc_vals,
                barrier=np.array([barrier2], dtype=float),
            )

            # measure_variance_collapse(
            #     prev_model=past_Classifier,
            #     curr_model=current_Classifier,
            #     interp_model=C,
            #     dataloader=train_loader, # Uses current task data (Memory Free)
            #     device=device,
            #     layer_name='block2', # Checking the output of the second conv block
            #     alpha=0.5 # Approximate alpha for expectation calculation
            # )
            # # -----------------------------    

        past_Classifier = deepcopy(C)
        print("task", task)
        print("\nTesting new model")

        # ----------------------------
        # Fill R[task, j] for all j
        # R[i,j] = acc(task j) after finishing training task i
        # ----------------------------
        C.eval()

        for j in range(T):
            # 1) Prepare a model that has output dim covering task j
            if j <= task:
                model_eval = C
                n_class_eval = n_class_for_task(task)  # current model output
            else:
                # For forward transfer (i<j), we need a model with outputs for future classes.
                # We DO NOT train those outputs; they remain randomly initialized.
                model_eval = deepcopy(C)
                model_eval = model_eval.expand_output_layer(config.init_classes, config.n_inc, j)
                model_eval.to(device)
                model_eval.eval()
                n_class_eval = n_class_for_task(j)

            # 2) Build task-j test loader (scaled with CURRENT scaler)
            X_test_j, Y_test_j = get_iter_test_dataset_task(
                X_test, Y_test,
                init_classes=config.init_classes,
                n_inc=config.n_inc,
                task_id=j
            )

            test_loader_j, _ = get_dataloader(
                X_test_j, Y_test_j,
                batchsize=config.batchsize,
                n_class=n_class_eval,
                scaler=scaler,
                train=False
            )

            # 3) Evaluate and store in R as fraction in [0,1]
            with torch.no_grad():
                acc_pct = test(config, model_eval, test_loader_j)  # returns 0..100
            R[task, j] = acc_pct / 100.0

        # Compute metrics on the prefix up to current task
        R_prefix = R[:task+1, :task+1]
        m = compute_cl_metrics(R_prefix)

        print(
            f"[CL metrics @task={task}] "
            f"ACC={m['ACC']*100:.2f}  "
            f"FWT={m['FWT']*100:.2f}  "
            f"BWT={m['BWT']*100:.2f}  "
            f"BWT+={m['BWT+']*100:.2f}  "
            f"F={m['F']*100:.2f}  "
            f"REM={m['REM']*100:.2f}"
        )


        with torch.no_grad():
            accuracy = test(config, C, test_loader)

            ls_a.append(accuracy)

        print("Task", task, "done.\n")


    final_metrics = compute_cl_metrics(R)
    print("\n==== Final Continual Learning Metrics (from full R) ====")
    print(
        f"ACC={final_metrics['ACC']*100:.2f}  "
        f"FWT={final_metrics['FWT']*100:.2f}  "
        f"BWT={final_metrics['BWT']*100:.2f}  "
        f"BWT+={final_metrics['BWT+']*100:.2f}  "
        f"F={final_metrics['F']*100:.2f}  "
        f"REM={final_metrics['REM']*100:.2f}"
    )

    # Optional: save R
    np.save("R_matrix.npy", R)

    report_result(config, ls_a)

if __name__ == "__main__":
    main()