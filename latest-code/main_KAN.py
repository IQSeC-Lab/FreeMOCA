import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
import joblib
from function import test, get_dataloader, class_pick_rand

from torch.utils.data import TensorDataset
from dynaconf import Dynaconf
from arguments import _parse_args
from setting import configurate, torch_setting
from _data import dataset
from train import data_task,data_task_joint, report_result
from train import run_batch_BCE as run_batch
import subprocess
import random
from models import KANClassifier, interpolate_models_KAN



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
C = KANClassifier()
C.to(device)

C_optimizer = optim.Adam(C.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

scaler = StandardScaler()

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

    for epoch in range(config.epochs):
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
        print(f"\nInterpolating task {task-1} and task {task}")
        C = interpolate_models_KAN(past_Classifier, current_Classifier, alpha=0.5, method="linear")
        C.to(device)

    past_Classifier = deepcopy(C)

    print("\nTesting new model")
    with torch.no_grad():
        accuracy = test(config, C, test_loader)

        ls_a.append(accuracy)

    print("Task", task, "done.\n")

report_result(config, ls_a)
