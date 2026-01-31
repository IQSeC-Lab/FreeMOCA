import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from dynaconf import Dynaconf
from arguments import _parse_args
from setting import configurate, torch_setting

from function import (
    test, get_dataloader, class_pick_rand,
    get_iter_test_dataset_task
)
from train import (
    data_task,
    data_task_domain,
    report_result,
    run_batch_BCE as run_batch
)
from models import Classifier, interpolate_models
from metrics_cl import compute_cl_metrics

from data_ import dataset
# ============================================================
# Setup
# ============================================================

config = Dynaconf()
args = _parse_args()
configurate(args, config)
torch_setting(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert config.scenario in ["class", "domain"], \
    "config.scenario must be 'class' or 'domain'"

# ============================================================
# Dataset
# ============================================================

X_train, Y_train, X_test, Y_test = dataset(config)

if config.scenario == "class":
    Y_train, Y_test = class_pick_rand(config, Y_train, Y_test)


# ============================================================
# Model & optimizer
# ============================================================

C = Classifier(config).to(device)
optimizer = optim.SGD(
    C.parameters(),
    lr=config.lr,
    momentum=config.momentum,
    weight_decay=config.weight_decay
)
criterion = nn.CrossEntropyLoss()
scaler = StandardScaler()

# ============================================================
# Helpers
# ============================================================

def n_class_for_task(task_id):
    if config.scenario == "domain":
        return config.init_classes
    return config.init_classes + task_id * config.n_inc


# ============================================================
# CL metric storage (class-IL only)
# ============================================================


T = config.nb_task
R = np.full((T, T), np.nan, dtype=float)

ls_a = []
past_Classifier = None

# ============================================================
# Training loop
# ============================================================

for task in range(config.nb_task):
    config.task = task
    config.n_class = n_class_for_task(task)

    # ----------------------------
    # Data
    # ----------------------------
    if config.scenario == "class":
        Xtr, Ytr, train_loader, Xte, Yte, test_loader, scaler = \
            data_task(config, X_train, Y_train, X_test, Y_test, scaler)
    else:
        Xtr, Ytr, train_loader, Xte, Yte, test_loader, scaler = \
            data_task_domain(config, X_train, Y_train, X_test, Y_test, scaler)

    config.nb_batch = len(Xtr) // config.batchsize

    # ----------------------------
    # Expand head (class-IL only)
    # ----------------------------
    if config.scenario == "class" and task > 0:
        C = C.expand_output_layer(
            config.init_classes,
            config.n_inc,
            task
        ).to(device)

    # ----------------------------
    # Train
    # ----------------------------
    for epoch in range(config.epochs):
        C.train()
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            run_batch(config, optimizer, C, criterion, inputs, labels)

    # ----------------------------
    # Interpolation
    # ----------------------------
    current_Classifier = deepcopy(C)

    if task > 0:
        C = interpolate_models(
            past_Classifier,
            current_Classifier,
            Lambda=None,
            method="linear",
            lambda_min=config.lambda_min,
            lambda_max=config.lambda_max
        ).to(device)

    past_Classifier = deepcopy(C)

    # ----------------------------
    # Evaluation (standard)
    # ----------------------------
    with torch.no_grad():
        acc = test(config, C, test_loader)
        ls_a.append(acc)


# ----------------------------
# Continual evaluation (ALL scenarios)
# ----------------------------
    C.eval()

    for j in range(task + 1):

        if config.scenario == "class":
            # ----- Class-IL evaluation -----
            if j <= task:
                model_eval = C
                n_eval = n_class_for_task(task)
            else:
                model_eval = deepcopy(C)
                model_eval = model_eval.expand_output_layer(
                    config.init_classes, config.n_inc, j
                ).to(device).eval()
                n_eval = n_class_for_task(j)

            Xte_j, Yte_j = get_iter_test_dataset_task(
                X_test, Y_test,
                config.init_classes, config.n_inc, j
            )

            test_loader_j, _ = get_dataloader(
                Xte_j, Yte_j,
                batchsize=config.batchsize,
                n_class=n_eval,
                scaler=scaler,
                train=False
            )

        else:
            # ----- Domain-IL evaluation -----
            Xte_j = X_test[j]
            Yte_j = Y_test[j]

            test_loader_j, _ = get_dataloader(
                Xte_j, Yte_j,
                batchsize=config.batchsize,
                n_class=config.init_classes,  # fixed label space
                scaler=scaler,
                train=False
            )

            model_eval = C

        with torch.no_grad():
            acc_j = test(config, model_eval, test_loader_j)

        R[task, j] = acc_j / 100.0

# ============================================================
# Final
# ============================================================

final_m = compute_cl_metrics(R)
print("\n==== Final CL Metrics ====")
print(
    f"ACC={final_m['ACC']*100:.2f}  "
    f"BWT={final_m['BWT']*100:.2f}  "
    f"FWT={final_m['FWT']*100:.2f}"
)
np.save("R_matrix.npy", R)

report_result(config, ls_a)