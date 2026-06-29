
# import numpy as np
# import torch
# import torch.nn as nn
# import random


import os, sys
import numpy as np
import torch

def get_domain_acc_data(data_root, tasks, j):

    X_list = []
    Y_list = []

    for k in range(j + 1):
        domain = tasks[k]
        file_path = os.path.join(data_root, f"{domain}_Domain_AZ_Test_Transformed.npz")

        data = np.load(file_path)
        X = data["X_test"]
        Y = data["Y_test"]
        X_list.append(X)
        Y_list.append(Y)

    X_concat = np.concatenate(X_list, axis=0)
    Y_concat_np = np.concatenate(Y_list, axis=0)
    Y_concat = torch.tensor(Y_concat_np, dtype=torch.long)

    return X_concat, Y_concat


import numpy as np
import torch
import torch.nn as nn
import os
import sys

def get_domain_data_joint(data_root, domain_folder, tasks, j):
    "all training dataset for the joint setting"
    X_list = []
    Y_list = []

    for k in range(j + 1):
        domain = tasks[k]
        file_path = os.path.join(data_root, f"{domain}_Domain_AZ_Train_Transformed.npz")

        data = np.load(file_path)
        X = data["X_train"]
        Y = data["Y_train"]
        X_list.append(X)
        Y_list.append(Y)

    X_concat = np.concatenate(X_list, axis=0)
    Y_concat_np = np.concatenate(Y_list, axis=0)
    Y_concat = torch.tensor(Y_concat_np, dtype=torch.long)

    return X_concat, Y_concat


def get_domain_data(data_root, domain_folder, is_train=True):    
    # 1. Construct Path
    file_path = os.path.join(data_root, f"{domain_folder}_Domain_AZ_Train_Transformed.npz")

    # 2. Safety Check
    if not os.path.exists(file_path):
        print(f"\n[ERROR] File not found: {file_path}")
        sys.exit(1)

    # 3. Load NPZ
    try:
        data = np.load(file_path)
        X = data['X_train']
        Y = data['Y_train']
            
    except Exception as e:
        print(f"[ERROR] Failed to load keys from {file_path}: {e}")
        # Print available keys to help debug
        print(f"Available keys in file: {list(data.keys())}")
        sys.exit(1)

    # 4. Convert Labels to LongTensor
    Y = torch.tensor(Y, dtype=torch.long)
    
    return X, Y

def oh(Y, num_classes):
    Y = torch.FloatTensor(Y)
    Y_oh = nn.functional.one_hot(Y.to(torch.int64), num_classes=num_classes)
    return Y_oh