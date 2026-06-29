import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from data_ import oh
import time
import copy


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

    print("class_pick_rand")
    print(class_arr)
    return Y_train, Y_test


# 0~49, 50~54, ... 95~99
def get_iter_train_dataset(x, y, n_class=None, n_inc=None, task=None):
   if task is not None:
    if task == 0:
       selected_indices = np.where(y < n_class)[0] 
    else:
       start = n_class - n_inc
       end = n_class
       selected_indices = np.where((y >= start) & (y < end))
    
    return x[selected_indices], y[selected_indices]


# 0~49, 0~54, ... 0~100 (acc 계산용)
def get_iter_acc_test_dataset(x, y, n_class):
    selected_indices = np.where(y < n_class)[0] 
    return x[selected_indices], y[selected_indices]

# 0~49, 50~54, ... 95~99 (BWT 계산용)
def get_iter_bwt_test_dataset(x, y, n_class=None, n_inc=None, task=None):
   if task is not None:
    if task == 0:
       selected_indices = np.where(y < n_class)[0] 
    else:
       start = n_class - n_inc
       end = n_class
       selected_indices = np.where((y >= start) & (y < end))
    return x[selected_indices], y[selected_indices]


def get_dataloader(x, y, batchsize, n_class, scaler, train = True):
    y_ = np.array(y, dtype=int)

    if train: 
        class_sample_count = np.array([len(np.where(y_ == t)[0]) for t in np.unique(y_)])
        weight = 1. / class_sample_count
        weight = 1. / class_sample_count
        min_ = (min(np.unique(y_)))
        samples_weight = np.array([weight[t-min_] for t in y_])
    
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    

    x_ = torch.from_numpy(x).type(torch.FloatTensor)
    y_ = torch.from_numpy(y_).type(torch.FloatTensor)

    # Scaling
    if train: scaler = scaler.partial_fit(x_)
    x_ = scaler.transform(x_)
    x_ = torch.FloatTensor(x_)
    
    # One-hot Encoding
    #MARK: 항상 (1, 100) 원핫
    y_oh = oh(y_, num_classes=100)
    y_oh = torch.Tensor(y_oh)

    data_tensored = torch.utils.data.TensorDataset(x_, y_oh)
    if train: Loader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize, num_workers=1, sampler=sampler)
    else: Loader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize)
    
    return Loader, scaler


# 여기서는 누적 합인 test 데이터가 들어옴
def test_acc(config, model, test_loader, allowed_classes):
    model.eval()
    correct = 0
    total = 0

    allowed_classes = list(range(allowed_classes))
    allowed_classes = torch.tensor(allowed_classes, device=config.device, dtype=torch.long)

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(config.device))

            # 마스킹해서 범위 내만 계산하게
            mask = torch.full_like(outputs, float('-inf'))
            mask[:, allowed_classes] = 0.0
            masked_outputs = outputs + mask
            
            _, predicted = torch.max(masked_outputs, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            labels = labels.to(config.device)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy for ACC: {accuracy * 100:.2f}%')
    return accuracy*100


# 여기서는 split된 test 데이터가 들어옴
def test_bwt(config, model, test_loaders):
    model.eval()
    accuracy_per_task = []
    with torch.no_grad():
        for task, test_loader in enumerate(test_loaders):
            allowed_classes = list(range(config.init_classes + task * config.n_inc))
            allowed_classes = torch.tensor(allowed_classes, device=config.device, dtype=torch.long)
            correct = 0
            total = 0
            
            for inputs, labels in test_loader:
                outputs = model(inputs.to(config.device))

                # 마스킹해서 범위 내만 계산하게
                mask = torch.full_like(outputs, float('-inf'))
                mask[:, allowed_classes] = 0.0
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