
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
import joblib
from models import Classifier
from function import test, get_dataloader, class_pick_rand
from torch.utils.data import TensorDataset
from dynaconf import Dynaconf
from arguments import _parse_args
from setting import configurate, torch_setting
from _data import dataset
from train import data_task, report_result
import subprocess
import random

# global variables and setting
config = Dynaconf()
args = _parse_args()
configurate(args, config)
torch_setting(config)


##############
# EMBER DATA #
##############

X_train, Y_train, X_test, Y_test = dataset(config)

# ############################################
# # data random arange #
# #############################################

Y_train, Y_test = class_pick_rand(config, Y_train, Y_test)


###############################
# Models and Hyper Parameters #
###############################
C = Classifier()

C.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C.to(device)

C_optimizer = optim.SGD(C.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss()

################################################
# sample selection and Batch training function # 
################################################

from train import run_batch_BCE as run_batch
###############################
# continual learning training #
###############################
ls_a = []
ls_a_old = []

scaler = StandardScaler()
x_ = torch.from_numpy(X_train).type(torch.FloatTensor)

print(f"before train")
for task in range(config.nb_task):
    config.n_class = config.init_classes + task * config.n_inc
    config.task = task

    X_train_t, Y_train_t, train_loader, X_test_t, Y_test_t, test_loader, scaler = data_task(config, X_train, Y_train, X_test, Y_test, scaler)
    config.nb_batch = int(len(X_train_t)/config.batchsize)
    # X_train_joint, Y_train_joint, train_loader_joint, X_test_joint, Y_test_joint, test_loader_joint, scaler = data_task_joint(config, X_train, Y_train, X_test, Y_test, scaler)

    if task > 0:
        C = C.expand_output_layer(config.init_classes, config.n_inc, task)
        C.to(device)

    for epoch in range(config.epochs):
        for n, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(config.device)
            labels = labels.float().to(config.device)
            C.train()
            run_batch(config, C, C_optimizer, criterion, BCELoss, inputs, labels)
    

    with torch.no_grad():
        print("test_new")
        accuracy = test(config, C, test_loader)
        ls_a.append(accuracy)
    print("task", task, "done")

report_result(config, ls_a)