# TAMIL-Ember/function.py

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from data_ import oh
import time
import copy

class Config:
    def __init__(self, seed, final_classes, init_classes, n_inc, device):
        self.seed_ = seed
        self.final_classes = final_classes
        self.init_classes = init_classes
        self.n_inc = n_inc
        self.device = device

def class_pick_rand(config, Y_train, Y_test):
    torch.manual_seed(config.seed_)

    class_arr = np.arange(config.final_classes)
    indices = torch.randperm(config.final_classes)
    class_arr = torch.index_select(torch.Tensor(class_arr), dim=0, index=indices)

    class_arr = np.array(class_arr)
    class_arr = list(class_arr)

    Y_train_ = copy.deepcopy(Y_train)
    Y_test_ = copy.deepcopy(Y_test)

    for i in range(0, config.final_classes):
        Y_train[np.where(Y_train_ == class_arr[i])] = i
        Y_test[np.where(Y_test_ == class_arr[i])] = i

    # print("class_pick_rand")
    # print(class_arr)
    return Y_train, Y_test, class_arr


def get_iter_train_dataset(x, y, n_class=None, n_inc=None, task=None):

    if task is not None:
        if task == 0:
            selected_indices = np.where(y < n_class)[0] 
        else:
            start = n_class - n_inc
            end = n_class
            selected_indices = np.where((y >= start) & (y < end))[0]
            return x[selected_indices], y[selected_indices]
    return x[selected_indices], y[selected_indices]

def get_iter_acc_test_dataset(x, y, n_class):
    selected_indices = np.where(y < n_class)[0] 
    return x[selected_indices], y[selected_indices]

def get_iter_bwt_test_dataset(x, y, n_class=None, n_inc=None, task=None):

    if task is not None:
        if task == 0:
            start = 0
            end = n_class
            selected_indices = np.where((y >= start) & (y < end))[0]
        else:
            start = n_class - n_inc
            end = n_class
            selected_indices = np.where((y >= start) & (y < end))[0]
        
        print(f"[get_iter_bwt_test_dataset] Task {task}: classes {start}-{end-1} ({len(selected_indices)} samples)")
        return x[selected_indices], y[selected_indices]
    
    return x, y


def get_dataloader(x, y, batchsize, n_class, scaler, train = True):

    y_ = np.array(y, dtype=int)

    
    if train: 
        class_sample_count = np.array([len(np.where(y_ == t)[0]) for t in np.unique(y_)])
        weight = 1. / class_sample_count
        weight = 1. / class_sample_count
        min_ = int((min(np.unique(y_))))
        samples_weight = np.array([weight[t-min_] for t in y_])
    
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    

    x_ = torch.from_numpy(x).type(torch.FloatTensor)

    # Scaling
    if train: 
        scaler = scaler.partial_fit(x_)
    x_scaled = scaler.transform(x_)
    x_ = torch.FloatTensor(x_scaled)

    y_ = torch.from_numpy(y_).type(torch.LongTensor)
    data_tensored = torch.utils.data.TensorDataset(x_, y_)

    if train: 
        Loader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize, num_workers=1, sampler=sampler)
    else: 
        Loader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize, shuffle=False)
    
    return Loader, scaler


def test_acc(config, model, test_loader, allowed_classes):

    model.eval()
    correct = 0
    total = 0

    allowed_classes_list = list(range(allowed_classes))
    allowed_classes_tensor = torch.tensor(allowed_classes_list, device=config.device, dtype=torch.long)

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(config.device))

            mask = torch.full_like(outputs, float('-inf'))
            mask[:, allowed_classes_tensor] = 0.0
            masked_outputs = outputs + mask
            
            _, predicted = torch.max(masked_outputs, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            labels = labels.to(config.device)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy for ACC: {accuracy * 100:.2f}%')
    return accuracy*100



def test_bwt(config, model, test_loaders):

    model.eval()
    accuracy_per_task = []

    with torch.no_grad():
        for task, test_loader in enumerate(test_loaders):
            allowed_classes_list = list(range(config.init_classes + task * config.n_inc))
            allowed_classes_tensor = torch.tensor(allowed_classes_list, device=config.device, dtype=torch.long)
            correct = 0
            total = 0
            
            for inputs, labels in test_loader:
                outputs = model(inputs.to(config.device))

                # 마스킹해서 범위 내만 계산하게
                mask = torch.full_like(outputs, float('-inf'))
                mask[:, allowed_classes_tensor] = 0.0
                masked_outputs = outputs + mask

                _, predicted = torch.max(masked_outputs, 1)
                _, labels = torch.max(labels, 1)
                total += labels.size(0)
                labels = labels.to(config.device)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            accuracy_per_task.append(accuracy)
    
    print(f"Test Accuracy for BWT: {accuracy_per_task}")
    return accuracy_per_task

def test_acc_tamil(config, model, test_loader, allowed_classes):

    model.eval()
    correct = 0
    total = 0

    allowed_classes_list = list(range(allowed_classes))
    allowed_classes_tensor = torch.tensor(allowed_classes_list, device=config.device, dtype=torch.long)

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(config.device))

            # Masking
            mask = torch.full_like(outputs, float('-inf'))
            mask[:, allowed_classes_tensor] = 0.0
            masked_outputs = outputs + mask
            
            _, predicted = torch.max(masked_outputs, 1)
            
            total += labels.size(0)
            labels = labels.to(config.device)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy for ACC: {accuracy * 100:.2f}%')
    return accuracy * 100


def test_bwt_tamil(config, model, test_loaders):

    model.eval()
    accuracy_per_task = []
    
    with torch.no_grad():
        for task, test_loader in enumerate(test_loaders):
            allowed_classes_list = list(range(config.init_classes + task * config.n_inc))
            allowed_classes_tensor = torch.tensor(allowed_classes_list, device=config.device, dtype=torch.long)
            correct = 0
            total = 0
            
            for inputs, labels in test_loader:
                outputs = model(inputs.to(config.device))

                # Masking
                mask = torch.full_like(outputs, float('-inf'))
                mask[:, allowed_classes_tensor] = 0.0
                masked_outputs = outputs + mask

                _, predicted = torch.max(masked_outputs, 1)
                total += labels.size(0)
                labels = labels.to(config.device)
                correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            accuracy_per_task.append(accuracy)
    
    print(f"Test Accuracy for BWT: {accuracy_per_task}")
    return accuracy_per_task