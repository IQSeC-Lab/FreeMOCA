
import numpy as np
import torch
import torch.nn as nn
import random

def get_ember_train_data(data_dir):
    X_train = np.load(data_dir + '/X_train.npy')
    Y_train = np.load(data_dir + '/Y_train.npy')
    return X_train, Y_train

def get_ember_test_data(data_dir):
    X_test = np.load(data_dir + '/X_test.npy')
    Y_test = np.load(data_dir + '/Y_test.npy')
    return X_test, Y_test

def shuffle_data(x_, y_, s):
    random.seed(s)
    indices = list(range(len(x_)))
    random.shuffle(indices)
    x_ = x_[indices]
    y_ = y_[indices]
    return x_, y_

def oh(Y, num_classes):
    Y = torch.FloatTensor(Y)
    Y_oh = nn.functional.one_hot(Y.to(torch.int64), num_classes=num_classes)
    return Y_oh
