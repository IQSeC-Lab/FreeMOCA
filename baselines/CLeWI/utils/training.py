# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.mlflow_logger import MLFlowLogger
from utils.status import ProgressBar

from utils.metrics import compute_fwt, compute_bwt, compute_forgetting, compute_rem_and_bwt_plus


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int, batch_idx: int) -> torch.Tensor:
    first = 50
    inc = 5
    end = first + k * inc
    masked_outputs = outputs.clone()
    masked_outputs[:, end:] = -float('inf')
    if batch_idx==0: print(f"## STEP 2-1. mask_classes. Task {k}: mask after classes: [{end}, 100)")
    return masked_outputs

def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, task_id=None):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    num_tasks_to_eval = len(dataset.test_loaders)
    
    # print("############## DEBUG test loaders")
    # print(f"Number of test_loaders: {len(dataset.test_loaders)}") 
    # print(f"Tasks to evaluate: 0 to {num_tasks_to_eval - 1}")
    # print(f"Current task (dataset.i): {dataset.i}")
    
    # for k in range(num_tasks_to_eval):
    #     test_loader = dataset.test_loaders[k]
    #     print(f"\nTask {k}:")
    #     print(f"  - Loader exists: {test_loader is not None}")
    #     print(f"  - Dataset size: {len(test_loader.dataset) if test_loader else 'N/A'}")
    #     print(f"  - Num batches: {len(test_loader) if test_loader else 'N/A'}")
    # print(f"{'='*50}\n")

    for k in range(num_tasks_to_eval):
        test_loader = dataset.test_loaders[k]
        print(f"## INIT: Evaluating on task {k}, loader has {len(test_loader)} batches.")
        if last and k < len(dataset.test_loaders) - 1:
            continue
        
        # test_loader = dataset.test_loaders[k]
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        
        batch_cnt = 0
        for batch_idx, data in enumerate(test_loader):
            batch_cnt += 1
            with torch.no_grad():
                inputs, labels = data
                # print(f"Task {k}: label range = {labels.min().item()} ~ {labels.max().item()}")
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                if batch_idx==0: print(f"## STEP 1. evaluate. outputs shape: {outputs.shape}, labels shape: {labels.shape}")
                # _, pred = torch.max(outputs.data, 1)
                # correct += torch.sum(pred == labels).item()

                if dataset.SETTING == 'class-il':
                    total += labels.shape[0]
                    # if batch_idx==0: print(f"## STEP 2. BEFORE masking - outputs[0, :10]: {outputs[0, :10].detach().cpu().tolist()}")
                    if batch_idx==0: print(f"## STEP 2. evaluate. BEFORE masking - argmax: {outputs[0].argmax().item()}, max value: {outputs[0].max().item():.4f}")
                    
                    outputs = mask_classes(outputs, dataset, k, batch_idx)
                    
                    # if batch_idx==0: print(f"## STEP 3. AFTER masking - outputs[0, :10]: {outputs[0, :10].detach().cpu().tolist()}")
                    if batch_idx==0: print(f"## STEP 2-3. evaluate. AFTER masking - argmax: {outputs[0].argmax().item()}, max value: {outputs[0].max().item():.4f}")
                    
                    _, pred = torch.max(outputs, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
                    if batch_idx==0: print("## STEP 3. pred[:10] =", pred[:10].detach().cpu().tolist(), "labels[:10] =", labels[:10].detach().cpu().tolist())

        print(f"Task {k}: Processed {batch_cnt} batches, {total} samples.")
        # print(f"Task {k}: Correct={correct}, Total={total}, Acc={correct/ total * 100 if total > 0 else 0:.2f}%")
        # accs.append((correct / total * 100 if total > 0 else 0)
        #     if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100 if total > 0 else 0)

    model.net.train(status)
    # print(f'Evaluation results (Class-IL): {accs_mask_classes}')
    return accs_mask_classes


def evaluate_cumulative(model: ContinualModel, dataset: ContinualDataset, current_task: int):
    """
    누적 evaluation - 현재까지 학습한 모든 task의 test set 합쳐서 평가 (Accuracy용)
    """
    status = model.net.training
    model.net.eval()
    
    correct_mask_classes, total = 0.0, 0.0
    
    for k in range(current_task + 1):
        test_loader = dataset.test_loaders[k]
        for batch_idx, data in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                # _, pred = torch.max(outputs.data, 1)
                # correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    if batch_idx==0: print(f"## STEP 2. evaluate_cumulative. BEFORE masking - argmax: {outputs[0].argmax().item()}, max value: {outputs[0].max().item():.4f}")

                    outputs = mask_classes(outputs, dataset, current_task, batch_idx)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
                    if batch_idx==0: print("## STEP 3. evaluate_cumulative. pred[:10] =", pred[:10].detach().cpu().tolist(), "labels[:10] =", labels[:10].detach().cpu().tolist())
    
    cumulative_acc = correct_mask_classes / total * 100 if total > 0 else 0
    
    model.net.train(status)
    return cumulative_acc


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    model.net.to(model.device)
    results, results_mask_classes = [], []
    cumulative_accs = []
    accuracy_matrix = []

    if not args.disable_log and not args.debug:
        logger = MLFlowLogger(dataset.SETTING, dataset.NAME, model.NAME,
                              experiment_name=args.experiment_name, parent_run_id=args.parent_run_id, run_name=args.run_name)
        logger.log_args(args.__dict__)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics and not args.debug:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        # if model.NAME != 'icarl' and model.NAME != 'pnn':
        #     random_results_task = evaluate(model, dataset_copy, task_id=dataset_copy.N_TASKS-1)

    if os.path.exists('old_model.pt'):
        os.remove('old_model.pt')
    if os.path.exists('net.pt'):
        os.remove('net.pt')

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        
        print(f"\n{'='*50}")
        print(f"STARTING TASK {t}")
        print(f"Train loader size: {len(train_loader.dataset)}")
        print(f"Test loaders available: {len(dataset.test_loaders)}")
        print(f"Current task index (dataset.i): {dataset.i}")
        print(f"{'='*50}\n")

        
        if hasattr(model.net, 'classifier'):
            print(f"Task {t}: model output size = {model.net.classifier.out_features}")
        elif hasattr(model.net, 'fc'):
            print(f"Task {t}: model output size = {model.net.fc.out_features}")
        else:
            print(f"Task {t}: model architecture unknown")

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        # if t and not args.ignore_other_metrics and not args.debug:
        #     accs = evaluate(model, dataset, last=True)
        #     results[t-1] = results[t-1] + accs[0]
        #     if dataset.SETTING == 'class-il':
        #         results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue

            for i, data in enumerate(train_loader):
              if args.debug and i > 3:
                  break

              # Helper to route the batch into inputs, labels, not_aug_inputs, optional logits
              logits = None
              # Case: dataloader returns 4 items (inputs, labels, not_aug_inputs, logits)
              if hasattr(dataset.train_loader.dataset, 'logits'):
                  # models/datasets that provide logits often return 4-tuple, but be robust:
                  if len(data) == 4:
                      inputs, labels, not_aug_inputs, logits = data
                  elif len(data) == 3:
                      inputs, labels, not_aug_inputs = data
                      logits = None
                  elif len(data) == 2:
                      inputs, labels = data
                      not_aug_inputs = inputs
                      logits = None
                  else:
                      raise ValueError(f"Unexpected batch size {len(data)} from dataloader (expected 2/3/4 tuples)")
                  # move to device
                  inputs = inputs.to(model.device)
                  labels = labels.to(model.device)
                  not_aug_inputs = not_aug_inputs.to(model.device)
                  if logits is not None:
                      logits = logits.to(model.device)
                      loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                  else:
                      loss = model.meta_observe(inputs, labels, not_aug_inputs)

              else:
                  # Default path: dataset doesn't expose logits.
                  # Accept 2- or 3-tuples; if 2-tuple, use inputs as not_aug_inputs (no augmentation)
                  if len(data) == 3:
                      inputs, labels, not_aug_inputs = data
                  elif len(data) == 2:
                      inputs, labels = data
                      not_aug_inputs = inputs  # fallback: no-augmentation input
                  else:
                      raise ValueError(f"Unexpected batch size {len(data)} from dataloader (expected 2 or 3 tuples)")

                  inputs = inputs.to(model.device)
                  labels = labels.to(model.device)
                  not_aug_inputs = not_aug_inputs.to(model.device)

                  loss = model.meta_observe(inputs, labels, not_aug_inputs)

              assert not math.isnan(loss)
              progress_bar.prog(i, len(train_loader), epoch, t, loss)


            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        # if 'clewi' in model.NAME and os.path.exists('old_model.pt') and os.path.exists('net.pt'):
        #     logger.log_artifact('old_model.pt', f'old_model_task_{t}')
        #     logger.log_artifact('net.pt', f'net_model_task_{t}')
        
        print(f"\n{'='*50}")
        print(f"############## EVALUATION for Task {t}, Test loaders count: {len(dataset.test_loaders)}")
        accs = evaluate(model, dataset, task_id=t)
        results.append(accs)
        print(f"BWT for Task {t}: {accs}\nBWT Matrix for Task {t}: {results}")

        cumulative_acc_value = evaluate_cumulative(model, dataset, t)
        cumulative_accs.append(cumulative_acc_value)
        print(f"ACC for Task {t}: {cumulative_acc_value}\nACC Matrix for Task {t}: {cumulative_accs}")
        print(f"{'='*50}\n")