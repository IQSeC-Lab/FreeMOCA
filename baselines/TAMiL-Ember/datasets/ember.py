# TAMIL-Ember/datasets/ember.py

import numpy as np
import torch
import copy
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import os


def get_ember_train_data(data_dir):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    Y_train = np.load(os.path.join(data_dir, 'Y_train.npy'))
    return X_train, Y_train

def get_ember_test_data(data_dir):
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    Y_test = np.load(os.path.join(data_dir, 'Y_test.npy'))
    return X_test, Y_test

def shuffle_data(x_, y_, s):
    """
    데이터 셔플 (seed 고정)
    """
    np.random.seed(s)
    indices = list(range(len(x_)))
    np.random.shuffle(indices)
    x_ = x_[indices]
    y_ = y_[indices]
    return x_, y_

def class_pick_rand(Y_train, Y_test, final_classes, seed):
    """
    클래스 순서 무작위 섞기 (seed 고정)
    
    Args:
        Y_train: 학습 데이터 레이블
        Y_test: 테스트 데이터 레이블
        final_classes: 최종 클래스 수 (100)
        seed: 랜덤 시드 값
    
    Returns:
        Y_train: 섞인 학습 데이터 레이블
        Y_test: 섞인 테스트 데이터 레이블
        class_arr: 원래 클래스 순서 배열
    """
    torch.manual_seed(seed)

    original_classes = np.unique(Y_train)
    print(f"Original unique classes: {original_classes}")
    print(f"Number of unique classes : {len(original_classes)}")

    if len(original_classes) != final_classes:
        print("The number of unique classes in Y_train does not match final_classes.")
        final_classes = len(original_classes)

    indices = torch.randperm(len(original_classes))
    class_order = original_classes[indices.numpy()]
    print(f"Shuffled class order (first 20): {class_order[:20]}")
    print(f"Shuffled class order (last 20): {class_order[-20:]}")

    # 레이블 재할당
    Y_train_new = copy.deepcopy(Y_train)
    Y_test_new = copy.deepcopy(Y_test)

    for new_label, old_label in enumerate(class_order):
        Y_train_new[Y_train == old_label] = new_label
        Y_test_new[Y_test == old_label] = new_label

    print(f"\nRemapped Y_train range : {Y_train_new.min()} to {Y_train_new.max()}")
    print(f"Remapped Y_test range : {Y_test_new.min()} to {Y_test_new.max()}")
    print(f"Number of unique classes after remapping - Train: {len(np.unique(Y_train_new))}")
    print(f"Number of unique classes after remapping - Test: {len(np.unique(Y_test_new))}")
    
    return Y_train_new, Y_test_new, class_order

def load_shared_data(shared_data_dir='/home/hansohyun329/FreeMOCA/shared_data'):
    """
    준비된 공유 데이터 로드
    """
    X_train = np.load(os.path.join(shared_data_dir, 'X_train.npy'))
    Y_train = np.load(os.path.join(shared_data_dir, 'Y_train.npy'))
    X_test = np.load(os.path.join(shared_data_dir, 'X_test.npy'))
    Y_test = np.load(os.path.join(shared_data_dir, 'Y_test.npy'))
    class_order = np.load(os.path.join(shared_data_dir, 'class_order.npy'))

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape} (dtype: {X_train.dtype}, {Y_train.dtype})")
    print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    print(f"class_order: {class_order.shape}")

    return X_train, Y_train, X_test, Y_test, class_order

def get_train_task_data(X, Y, task_id, initial_classes, increment_classes):
    """
    현재 task의 "새 클래스"만 추출 (학습용)
    예시 :
        Task 0 : 클래스 0~49 (50개)
        Task 1 : 클래스 50~54 (5개)
        Task 2 : 클래스 55~59 (5개)
    """
    if task_id == 0:
        start = 0
        end = initial_classes
    else:
        start = initial_classes + (task_id - 1) * increment_classes
        end = initial_classes + task_id * increment_classes

    indices = np.where((Y>=start)&(Y<end))[0]
    print(f"[Train] Task {task_id}: classes {start}-{end-1} ({end-start} classes, {len(indices)} samples)")
    return X[indices], Y[indices]

def get_test_task_data(X, Y, task_id, initial_classes, increment_classes):
    """
    현재까지의 모든 task 데이터 추출 (테스트용)
    """
    n_classes = initial_classes + task_id * increment_classes
    indices = np.where(Y < n_classes)[0]
    print(f"[Test]  Task {task_id}: classes 0-{n_classes-1} ({n_classes} classes, {len(indices)} samples)")
    return X[indices], Y[indices]

def get_task_data(X, Y, n_class):
    """
    현재까지의 모든 task 데이터 추출 (테스트용)
    """
    indices = np.where(Y < n_class)[0]
    return X[indices], Y[indices]

def get_balanced_dataloader(X, Y, n_class, batch_size, scaler=None, train=True, num_workers=4, seed=42):
    """
    균형잡힌 DataLoader 생성 (WeightedRandomSampler 사용)
    
    Args :
        X : 특징 데이터 (numpy array)
        Y : 레이블 데이터 (numpy array)
        n_class : 현재까지의 클래스 수
        batch_size : 배치 크기
        scaler : StandardScaler, None이면 새로 생성
        train : 학습용 여부 (True면 샘플링 적용)
        num_workers : DataLoader worker 수
        seed : 랜덤 시드 값
    
    Returns :
        loader : DataLoader 객체
        scaler : StandardScaler 객체
    """
    Y_ = np.array(Y, dtype=int)

    # WeightedRandomSampler (학습 시에만)
    sampler = None
    if train:
        unique_classes = np.unique(Y_)

        # 클래스별 샘플 수 계산
        class_sample_count = {}
        for cls in unique_classes:
            class_sample_count[cls] = len(np.where(Y_ == cls)[0])

        # 샘플별 가중치 계산
        samples_weight = []
        for label in Y_:               
            weight = 1.0 / class_sample_count[label]
            samples_weight.append(weight)

        samples_weight = torch.FloatTensor(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    # Tensor 변환
    X_ = torch.from_numpy(X).type(torch.FloatTensor)
    Y_ = torch.from_numpy(Y_).type(torch.LongTensor)

    # Scaling
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_)

    X_scaled = scaler.transform(X_)
    X_scaled = np.clip(X_scaled, -10, 10)
    
    X_ = torch.FloatTensor(X_scaled)

    # Dataset
    dataset = TensorDataset(X_, Y_)

    # Worker seed 함수
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    # DataLoader
    if train:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler, 
            num_workers=num_workers, 
            pin_memory=True, 
            worker_init_fn=worker_init_fn, 
            drop_last=True)
    else:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True, 
            worker_init_fn=worker_init_fn, 
            drop_last=True)

    return loader, scaler