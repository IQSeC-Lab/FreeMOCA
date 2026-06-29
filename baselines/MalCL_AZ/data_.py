
import numpy as np
import torch
import torch.nn as nn
import random

def get_az_train_data(data_dir):
    data = np.load(data_dir + '/AZ_Class_Train.npz')
    X_train = data['X_train']
    Y_train = data['Y_train']
    return X_train, Y_train

def get_az_test_data(data_dir):
    data = np.load(data_dir + '/AZ_Class_Test.npz')
    X_test = data['X_test']
    Y_test = data['Y_test']
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
