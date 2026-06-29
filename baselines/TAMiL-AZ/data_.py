
import numpy as np
import torch
import torch.nn as nn
import random

def get_selected_classes(target_classes):
    classes_Y = [i for i in range(100)]
    selected_classes = np.random.choice(classes_Y, target_classes,replace=False)
    return selected_classes

def get_continual_AZ_class_data(data_dir, num_classes, train=True):
    
    if train:
        data_dir = data_dir + '/'
        XY_train = np.load(data_dir + 'AZ_Class_Train.npz')
        X_tr, Y_tr = XY_train['X_train'], XY_train['Y_train']

        return X_tr, Y_tr
    else:
        data_dir = data_dir + '/'
        XY_test = np.load(data_dir + 'AZ_Class_Test.npz')
        X_test, Y_test = XY_test['X_test'], XY_test['Y_test']

        return X_test, Y_test


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
