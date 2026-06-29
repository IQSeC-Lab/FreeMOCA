
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from models import Generator, Discriminator, Classifier
from function import test_acc, test_bwt, class_pick_rand
from metric import compute_fwt, compute_bwt, compute_rem_and_bwt_plus, compute_forgetting

from torch.utils.data import TensorDataset
from dynaconf import Dynaconf
from sample_selection import L2_One_Hot, L1_B_Mean, L1_C_Mean
from arguments import _parse_args
from setting import configurate, torch_setting
from _data import dataset
from train import data_task, report_result, mean_logits, collect_logits, col_arr
import subprocess
import random


def main():
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

  #MARK: 다른 실험들과 동일하게 유지
  # Y_train, Y_test = class_pick_rand(config, Y_train, Y_test)


  ###############################
  # Models and Hyper Parameters #
  ###############################

  G = Generator()
  D = Discriminator()
  C = Classifier()

  G.train()
  D.train()
  C.train()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  G.to(device)
  D.to(device)
  C.to(device)

  G_optimizer = optim.Adam(G.parameters(), lr=config.lr)
  D_optimizer = optim.Adam(D.parameters(), lr=config.lr)
  C_optimizer = optim.SGD(C.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

  criterion = nn.CrossEntropyLoss()
  BCELoss = nn.BCELoss()

  ################################################
  # sample selection and Batch training function # 
  ################################################

  if config.sample_select == 'L2_One_Hot' : from sample_selection import common_vars, L2_One_Hot as get_replay_with_label
  elif config.sample_select == 'L1_B_Mean' : from sample_selection import common_vars, L1_B_Mean as get_replay_with_label
  elif config.sample_select == 'L1_C_Mean' : from sample_selection import common_vars, L1_C_Mean as get_replay_with_label

  if config.Generator_loss == 'FML' : from train import run_batch_FML as run_batch
  elif config.Generator_loss == 'BCE' : from train import run_batch_BCE as run_batch

  ###############################
  # continual learning training #
  ###############################
  ls_a = []
  G.reinit()
  D.reinit()


  scaler = StandardScaler()
  x_ = torch.from_numpy(X_train).type(torch.FloatTensor)


  ###############################
  #MARK: 미리 데이터 다 뽑는 걸로 변경
  print(f"prepare train/test dataset")
  X_train_ts = []
  train_loaders = []
  acc_test_loaders = []
  bwt_test_loaders = []
  metrix_test_loaders = []

  for task in range(config.nb_task):
    config.n_class = config.init_classes + task * config.n_inc
    config.task = task
    X_train_t, Y_train_t, X_test_acc_t, Y_test_acc_t, X_test_bwt_t, Y_test_bwt_t, train_loader, acc_test_loader, bwt_test_loader, scaler = data_task(config, X_train, Y_train, X_test, Y_test, scaler)
    # print(f"task {task}, Y_train_t.shape {Y_train_t.shape}, Y_test_acc_t.shape {Y_test_acc_t.shape}")
    X_train_ts.append(X_train_t)
    train_loaders.append(train_loader)
    acc_test_loaders.append(acc_test_loader)
    bwt_test_loaders.append(bwt_test_loader)

  #MARK: 매트릭스 만들기
  print(f"make matrix")
  tasks = [f"task {i}" for i in range(1, config.nb_task+1)]
  rows = ([f"after task {i}" for i in range(1, config.nb_task+1)])
  accuracy_matrix = pd.DataFrame(0.0, index=rows, columns=tasks)
  print(accuracy_matrix)


  print(f"before train")
  for task in range(config.nb_task):
    config.n_class = config.init_classes + task * config.n_inc
    config.task = task
    X_train_t = X_train_ts[task]
    train_loader = train_loaders[task]
    acc_test_loader = acc_test_loaders[task]

    config.nb_batch = int(len(X_train_t)/config.batchsize)
    logits_collect = col_arr(config, X_train_t)

    #MARK: 항상 100개 output
    # if task > 0:
    #   C = C.expand_output_layer(config.init_classes, config.n_inc, task)
    #   C.to(device)

    for epoch in range(config.epochs):
      for n, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.float()
        labels = labels.float()
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        # 디버깅 용 start
        # if epoch == 0 and n == 0:
        #   # task 0: 0~49, task 1: 50~54, task 2: 55~59 ...
        #   print("training labels shape:", labels.shape)
        #   print("training labels argmax min/max:",
        #     labels.argmax(dim=1).min().item(),
        #     labels.argmax(dim=1).max().item())
        # 디버깅 용 end

        if config.task > 0:
          synthetic, pred_label, logits_gen = common_vars(config, past_Generator, past_Classifier)
          replay, re_label = get_replay_with_label(config, synthetic, pred_label, logits_gen, logits_real)
          inputs=torch.cat((inputs,replay),0)
          labels=torch.cat((labels,re_label),0)

          # 디버깅 용 start
          # if epoch == 0 and n == 0:
          #   # task 1: 0~49, task 2: 0~54, task 3: 0~59 ...
          #   print("re_label labels shape:", labels.shape)
          #   print("re_label labels argmax min/max:",
          #     re_label.argmax(dim=1).min().item(),
          #     re_label.argmax(dim=1).max().item())
          #   # task 1: 0~54, task 2: 0~59, task 3: 0~64
          #   print("labels cat shape (after):", labels.shape)
          #   print("labels cat argmax min/max:",
          #     labels.argmax(dim=1).min().item(),
          #     labels.argmax(dim=1).max().item())
          # 디버깅 용 end

        C.train()
        G.train()
        D.train()
        run_batch(config, G, D, C, G_optimizer, D_optimizer, C_optimizer, criterion, BCELoss, inputs, labels)
        logits_collect = collect_logits(config, C, logits_collect, inputs, labels, n)
        print("\r", task, "task", epoch+1, "epoch", n, "/", config.nb_batch ,"batch", end="")
    print("\n")

    # past
    past_Generator = deepcopy(G)
    past_Classifier = deepcopy(C)
    logits_real = mean_logits(config, logits_collect)

    # 각 테스크마다
    with torch.no_grad():
      print(f"test_new, allowed_classes for ACC: {config.n_class}")
      #MARK: ACC 계산용
      accuracy = test_acc(config, C, acc_test_loader, allowed_classes=config.n_class)
      ls_a.append(accuracy)
      #MARK: BWT 계산용
      accuracy_per_task = test_bwt(config, C, bwt_test_loaders)
      accuracy_matrix.loc[f"after task {task+1}", :] = accuracy_per_task
    print("task", task, "done")
  
  report_result(config, ls_a)

  print("Accuracy matrix, based on all classes up to the task being evaluated")
  print(accuracy_matrix)
  R = accuracy_matrix.values
  FWT = compute_fwt(R)
  BWT = compute_bwt(R)
  F = compute_forgetting(R)
  REM, BWT_plus = compute_rem_and_bwt_plus(BWT)

  print(f"\n=> FWT = {FWT}")
  print(f"=> BWT = {BWT}")
  print(f"=>  F = {F}")
  print(f"=> REM = {REM}")
  print(f"=> BWT+ = {BWT_plus}")


if __name__ == "__main__":
    main()