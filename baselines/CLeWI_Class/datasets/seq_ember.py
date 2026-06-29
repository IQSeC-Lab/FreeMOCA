import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from backbone.ResNet18 import resnet18
from torch.utils.data import Subset, DataLoader
from sklearn.preprocessing import StandardScaler

class Classifier(nn.Module):
    def __init__(self, input_features=2381, output_dim=100, drop_prob=0.5):
        super(Classifier, self).__init__()
        print(f"############## DEBUG model Classifier")

        self.input_features = input_features
        self.output_dim = output_dim
        self.drop_prob = drop_prob

        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_features, 512, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 3, 3, 1),
            nn.BatchNorm1d(256),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.MaxPool1d(3, 3, 1)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.drop_prob),
            nn.ReLU()
        )
        
        self.fc1_f = nn.Flatten()
        self.fc1 = nn.Linear(128, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        self.fc1_drop1 = nn.Dropout(self.drop_prob)
        self.fc1_act1 = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        original_shape = x.size()

        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)

        x = x.view(batch_size, self.input_features, -1)
        x = self.block1(x)
        x = self.block2(x)

        x = self.fc1_f(x)
        x = self.fc1(x)
        x = self.fc1_bn1(x)
        x = self.fc1_drop1(x)
        x = self.fc1_act1(x)

        x = self.softmax(x)

        if len(original_shape) == 3:
            x = x.view(original_shape[0], original_shape[1], -1)

        return x

    def predict(self, x_data):
        return self.forward(x_data)

    def get_logits(self, x):
        original_shape = x.size()

        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)

        x = x.view(batch_size, self.input_features, -1)
        x = self.block1(x)
        x = self.block2(x)

        logits = self.fc1_f(x)
        return logits.detach()

def get_parser():
    from utils.args import ArgumentParser, add_management_args, add_experiment_args
    parser = ArgumentParser(description='Sequential EMBER')
    add_management_args(parser)
    add_experiment_args(parser)
    
    parser.add_argument('--sub_dataset', type=str, default='ember',
                        choices=['ember', 'az'],
                        help='Sub-dataset to use: ember or az')
    
    return parser

class MyEMBER(Dataset):
    """Simple Dataset wrapper for precomputed numpy feature arrays."""
    def __init__(self, features: np.ndarray, targets: np.ndarray, sub_dataset: str):
        self.data = features
        self.targets = targets.astype(int)
        self.sub_dataset = sub_dataset

    def __len__(self):
        return len(self.targets)

    # EMBER
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x = self.data[index]
        y = int(self.targets[index])
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # EMBER
        if self.sub_dataset=="ember":
            # pad from 2381 → 2401 (20 zeros)
            if x_tensor.shape[0] < 2401:
                pad_len = 2401 - x_tensor.shape[0]
                x_tensor = torch.cat([x_tensor, torch.zeros(pad_len, dtype=torch.float32)], dim=0)

            # reshape into 49x49
            x_tensor = x_tensor.view(49, 49)

            # add channel dimension → (1, 49, 49)
            x_tensor = x_tensor.unsqueeze(0)

            # expand to 3 channels → (3, 49, 49)
            x_tensor = x_tensor.repeat(3, 1, 1)
        
        # AZ
        elif self.sub_dataset=="az":            
            # pad from 2439 → 2500 (61 zeros)
            if x_tensor.shape[0] < 2500:
                pad_len = 2500 - x_tensor.shape[0]
                x_tensor = torch.cat([x_tensor, torch.zeros(pad_len, dtype=torch.float32)], dim=0)

            # reshape into 50x50
            x_tensor = x_tensor.view(50, 50)

            # add channel dimension → (1, 50, 50)
            x_tensor = x_tensor.unsqueeze(0)

            # expand to 3 channels → (3, 50, 50)
            x_tensor = x_tensor.repeat(3, 1, 1)

        return x_tensor, y


class SequentialEMBER(ContinualDataset):
    """
    EMBER dataset adapted to the repo interface with class-incremental split.
    Task 0: 50 classes
    Tasks 1-10: 5 classes each
    """
    NAME = 'seq_ember'
    SETTING = 'class-il'
    N_CLASSES = 100
    N_TASKS = 11
    N_CLASSES_PER_TASK = 5          # required by ContinualDataset (default per-task split)

    INITIAL_CLASSES = 50
    INCREMENTAL_CLASSES = 5
    def get_transform(self):
        
        """
        EMBER features are already numeric vectors; we don’t apply image-style
        transforms. Return identity.
        """
        # def transform(x):
        #     return x
        # return transform
        return None

    def get_examples_number(self):
        root = os.path.join(base_path(), 'top_classes_100')
        XY_train = np.load(os.path.join(root, 'XY_train.npz'))
        return XY_train['X_train'].shape[0]


    def get_data_loaders(self):
        # root = os.path.join(base_path(), 'top_classes_100')

        if not hasattr(self, 'train_loaders') or not self.train_loaders:

            # EMBER
            if self.args.sub_dataset=="ember":
                root = os.path.join(base_path(), 'EMBER_Class')
                XY_train = np.load(os.path.join(root, 'XY_train.npz')) 
                XY_test  = np.load(os.path.join(root, 'XY_test.npz'))
                scaler = StandardScaler()
                train_X, train_y = XY_train['X_train'], XY_train['Y_train']
                test_X, test_y   = XY_test['X_test'], XY_test['Y_test']
                train_X = scaler.fit_transform(train_X)
                test_X = scaler.transform(test_X)
                train_X = train_X.astype(np.float32)
                test_X = test_X.astype(np.float32)

            # AZ
            elif self.args.sub_dataset=="az":
                root = os.path.join(base_path(), 'AZ_Class')
                XY_train = np.load(os.path.join(root, 'AZ_Class_Train.npz')) 
                XY_test  = np.load(os.path.join(root, 'AZ_Class_Test.npz')) 
                train_X, train_y = XY_train['X_train'], XY_train['Y_train']
                test_X, test_y   = XY_test['X_test'], XY_test['Y_test']
                
            train_X = np.nan_to_num(train_X).astype('float32')
            test_X  = np.nan_to_num(test_X).astype('float32')

            self.n_features = train_X.shape[1]

            train_dataset = MyEMBER(train_X, train_y, self.args.sub_dataset)
            test_dataset  = MyEMBER(test_X, test_y, self.args.sub_dataset)

            # one-hot 
            train_y = torch.FloatTensor(train_y)
            train_y = nn.functional.one_hot(train_y.to(torch.int64), num_classes=100)

            test_y = torch.FloatTensor(test_y)
            test_y = nn.functional.one_hot(test_y.to(torch.int64), num_classes=100)

            if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset, None, self.NAME)

            # Build indices for each task according to schedule
            train_indices_per_task = self.split_by_schedule(train_dataset)
            test_indices_per_task  = self.split_by_schedule(test_dataset)

            self.train_indices_per_task = train_indices_per_task
            self.test_indices_per_task  = test_indices_per_task

            # Create loaders manually instead of using store_masked_loaders
            self.train_loaders = []
            self.test_loaders = []

            for task_id, (train_idx, test_idx) in enumerate(zip(train_indices_per_task, test_indices_per_task)):
                train_subset = Subset(train_dataset, train_idx)
                test_subset  = Subset(test_dataset, test_idx)

                # print(f"self.args.batch_size {self.args.batch_size}")

                train_loader = DataLoader(
                    train_subset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=64
                )
                test_loader = DataLoader(
                    test_subset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=64
                )

                self.train_loaders.append(train_loader)
                self.test_loaders.append(test_loader)

        # Start with first task
        self.train_loader = self.train_loaders[self.i] # self.train_loaders[0] -> self.i
        return self.train_loader, self.test_loaders[self.i] # self.test_loaders[0] -> self.i

    def split_by_schedule(self, dataset):
        """
        Splits dataset indices into tasks according to:
        - Task 0: INITIAL_CLASSES (0~49)
        - Tasks 1..: INCREMENTAL_CLASSES (50~54, 55~59, ...)
        """
        targets = np.array(dataset.targets)
        classes = np.unique(targets)

        assert len(classes) == self.N_CLASSES, f"Expected {self.N_CLASSES}, got {len(classes)}"
        n_tasks = self.args.n_tasks

        # Assign classes to tasks
        # task_classes = [class_order[:self.INITIAL_CLASSES]]
        task_classes = [list(range(0, self.INITIAL_CLASSES))] # Task 0 classes
        # for i in range(self.INITIAL_CLASSES, self.N_CLASSES, self.INCREMENTAL_CLASSES):
        #     task_classes.append(class_order[i:i + self.INCREMENTAL_CLASSES])
        for i in range(1, n_tasks):
            start_idx = self.INITIAL_CLASSES + (i - 1) * self.INCREMENTAL_CLASSES
            end_idx = start_idx + self.INCREMENTAL_CLASSES
            # task_classes.append(class_order[start_idx:end_idx])
            task_classes.append(list(range(start_idx, end_idx)))

        print("############## DEBUG class labels")
        for i, tc in enumerate(task_classes):
            print(f"Task {i}: classes {min(tc)}~{max(tc)} ({len(tc)} classes)")
        task_indices = [np.where(np.isin(targets, t_classes))[0] for t_classes in task_classes]

        for i, (task_cls, task_idx) in enumerate(zip(task_classes, task_indices)):
            actual_labels = targets[task_idx]
            print(f"Task {i}:")
            print(f"  Assigned classes: {min(task_cls)}~{max(task_cls)}")
            print(f"  Actual labels in data: {np.unique(actual_labels)}")
            print(f"  Sample count: {len(task_idx)}")
        return task_indices

    def get_n_classes_for_task(self, task_id: int):
        """Returns cumulative number of classes for a given task."""
        if task_id == 0:
            return self.INITIAL_CLASSES
        else:
            return self.INITIAL_CLASSES + self.INCREMENTAL_CLASSES * task_id

    def get_backbone(self, task_id: int = 0):
        """Return ResNet18 backbone with correct output classes for the task."""
        n_classes = 100 # self.get_n_classes_for_task(task_id) #task_id = 0 : default
        return resnet18(nclasses=n_classes, nf=int(64 * self.args.resnet_width))
    # def get_backbone(self, task_id: int = 0):

    #     """Return our custom 1D Conv Classifier for EMBER features."""
    #     n_classes = self.get_n_classes_for_task(task_id)
    #     in_features = getattr(self, "n_features", 2381)
    #     return Classifier(input_features=in_features, output_dim=n_classes)
    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 256

    @staticmethod
    def get_minibatch_size():
        return SequentialEMBER.get_batch_size()

    def get_scheduler(self, model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.optim_wd,
                                    momentum=args.optim_mom)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     model.opt,
        #     # milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)],
        #     milestones=[int(args.n_epochs * 0.7), int(args.n_epochs * 0.9)],
        #     gamma=0.1,
        #     verbose=False
        # )
        scheduler = torch.optim.lr_scheduler.MultiStepLR( model.opt,
          milestones=[int(args.n_epochs * 0.7), int(args.n_epochs * 0.9)],
          gamma=0.1
      )

        return scheduler