import os
import sys
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Utility
# ============================================================

def oh(Y, num_classes):
    Y = torch.FloatTensor(Y)
    return nn.functional.one_hot(Y.to(torch.int64), num_classes=num_classes)


# ============================================================
# CLASS-INCREMENTAL DATA
# ============================================================

def load_class_data_ember(data_dir):
    train_npz = np.load(os.path.join(data_dir, "XY_train.npz"))
    test_npz  = np.load(os.path.join(data_dir, "XY_test.npz"))

    X_train, Y_train = train_npz["X_train"], train_npz["Y_train"]
    X_test,  Y_test  = test_npz["X_test"],  test_npz["Y_test"]

    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test  = torch.tensor(Y_test,  dtype=torch.long)

    return X_train, Y_train, X_test, Y_test


def load_class_data_az(data_dir):
    train_npz = np.load(os.path.join(data_dir, "AZ_Class_Train.npz"))
    test_npz  = np.load(os.path.join(data_dir, "AZ_Class_Test.npz"))

    X_train, Y_train = train_npz["X_train"], train_npz["Y_train"]
    X_test,  Y_test  = test_npz["X_test"],  test_npz["Y_test"]

    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test  = torch.tensor(Y_test,  dtype=torch.long)

    return X_train, Y_train, X_test, Y_test


# ============================================================
# DOMAIN-INCREMENTAL DATA
# ============================================================

def load_domain_data_ember(data_root, domain, is_train=True):
    folder = os.path.join(data_root, domain)
    fname  = "XY_train.npz" if is_train else "XY_test.npz"
    path   = os.path.join(folder, fname)

    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {path}")
        sys.exit(1)

    data = np.load(path)

    if is_train:
        X, Y = data["X_train"], data["Y_train"]
    else:
        X, Y = data["X_test"], data["Y_test"]

    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


def load_domain_data_az(data_root, domain, is_train=True):
    subset = "Train" if is_train else "Test"
    fname  = f"{domain}_Domain_AZ_{subset}_Transformed.npz"
    path   = os.path.join(data_root, fname)

    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {path}")
        sys.exit(1)

    data = np.load(path)

    if is_train:
        if "X_train" in data:
            X, Y = data["X_train"], data["Y_train"]
        else:
            X, Y = data["x_train"], data["y_train"]
    else:
        if "X_test" in data:
            X, Y = data["X_test"], data["Y_test"]
        else:
            X, Y = data["x_test"], data["y_test"]

    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def dataset(config):
    """
    Returns:
        X_train, Y_train, X_test, Y_test

    For Domain-IL:
        X_train, Y_train and X_test, Y_test are LISTS
        indexed by task/domain.
    """

    # =====================================================
    # CLASS-INCREMENTAL
    # =====================================================
    if config.scenario == "class":
        if config.dataset == "EMBER":
            return load_class_data_ember(config.train_data)
        elif config.dataset == "AZ":
            return load_class_data_az(config.train_data)
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")

    # =====================================================
    # DOMAIN-INCREMENTAL
    # =====================================================
    elif config.scenario == "domain":
        X_train_all, Y_train_all = [], []
        X_test_all,  Y_test_all  = [], []

        for domain in config.domains:
            if config.dataset == "EMBER":
                Xtr, Ytr = load_domain_data_ember(
                    config.data_root, domain, is_train=True
                )
                Xte, Yte = load_domain_data_ember(
                    config.data_root, domain, is_train=False
                )

            elif config.dataset == "AZ":
                Xtr, Ytr = load_domain_data_az(
                    config.data_root, domain, is_train=True
                )
                Xte, Yte = load_domain_data_az(
                    config.data_root, domain, is_train=False
                )

            else:
                raise ValueError(f"Unknown dataset: {config.dataset}")

            X_train_all.append(Xtr)
            Y_train_all.append(Ytr)
            X_test_all.append(Xte)
            Y_test_all.append(Yte)

        return X_train_all, Y_train_all, X_test_all, Y_test_all

    else:
        raise ValueError(f"Unknown scenario: {config.scenario}")
