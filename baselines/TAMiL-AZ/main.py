# TAMIL-Ember/train.py (Scaler 문제 해결)

import torch
import sys
import numpy as np
import pandas as pd
import os
import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from models.tamil_ember import TAMIL_Ember
from models.buffer import Buffer
from models.tam import TAM
from models.tamil_ember import BackboneMLP
from data_ import get_ember_train_data, get_ember_test_data, get_continual_AZ_class_data
from function import (Config, class_pick_rand, get_iter_train_dataset, 
                    get_iter_acc_test_dataset, get_iter_bwt_test_dataset, 
                    get_dataloader, test_acc, test_bwt, test_acc_tamil, test_bwt_tamil)

class Args:
    buffer_size = 10000
    minibatch_size = 256
    reg_weight = 0.1
    beta = 0.5
    alpha = 0.1
    ema_update_freq = 0.05
    ema_alpha = 0.999
    use_pairwise_loss_after_ae = False
    pairwise_weight = 0.1
    load_best_args = False
    lr = 0.001

args = Args()

input_dim = 2439
hidden_dim = 512
feat_dim = 128
latent_dim = 64
code_dim = 64
n_tasks = 11
n_classes = 100

initial_classes = 50
increment_classes = 5
batch_size = 256
n_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

config = Config(
    seed=seed,
    final_classes=n_classes,
    init_classes=initial_classes,
    n_inc=increment_classes,
    device=device
)

log_file = './logs/train.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
print("="*70)
print("Logging to:", log_file)
print("="*70)

data_dir = "./data/AZ_Class"
torch.manual_seed(seed)
np.random.seed(seed)

print("\n[Data Loading]")
X_train, Y_train = get_continual_AZ_class_data(data_dir, n_classes, train=True)
X_test, Y_test = get_continual_AZ_class_data(data_dir, n_classes, train=False)

Y_train, Y_test, class_order = class_pick_rand(config, Y_train, Y_test)

print("\n[Model Initialization]")
backbone = BackboneMLP(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    feat_dim=feat_dim,
    n_tasks=n_tasks,
    code_dim=code_dim,
    n_classes=n_classes,
    ae_type='relu'
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
transform = None

model = TAM(backbone, loss_fn, args, transform).to(device)
print(f"Model: TAM")
print(f"Device: {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


def test_acc_tamil(config, model, test_loader, allowed_classes):

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


scaler = StandardScaler()
print("\n[Fitting scaler with all train data]")
scaler.fit(X_train)


print("\n[Preparing test loaders]")
acc_test_loaders = []
bwt_test_loaders = []

for task_id in range(n_tasks):
    n_class_temp = initial_classes + task_id * increment_classes
    
    X_te_acc, Y_te_acc = get_iter_acc_test_dataset(X_test, Y_test, n_class=n_class_temp)
    acc_loader, _ = get_dataloader(X_te_acc, Y_te_acc, n_class=n_classes, 
                                   scaler=scaler, batchsize=batch_size, train=False)
    acc_test_loaders.append(acc_loader)
    
    X_te_bwt, Y_te_bwt = get_iter_bwt_test_dataset(X_test, Y_test, 
                                                    n_class=n_class_temp,
                                                    n_inc=increment_classes, 
                                                    task=task_id)
    bwt_loader, _ = get_dataloader(X_te_bwt, Y_te_bwt, n_class=n_classes,
                                   scaler=scaler, batchsize=batch_size, train=False)
    bwt_test_loaders.append(bwt_loader)


ls_a = []
tasks = [f"task {i}" for i in range(1, n_tasks+1)]
rows = [f"after task {i}" for i in range(1, n_tasks+1)]
accuracy_matrix = pd.DataFrame(0.0, index=rows, columns=tasks)

print(f"\nmake matrix")
print(accuracy_matrix)


train_scaler = StandardScaler()

print(f"\nbefore train")
for task_id in range(n_tasks):
    print(f"\n==== Task {task_id} ====")
    
    config.n_class = initial_classes + task_id * increment_classes
    config.task = task_id
    
    X_tr_task, Y_tr_task = get_iter_train_dataset(
        X_train, Y_train, 
        n_class=config.n_class, 
        n_inc=increment_classes, 
        task=task_id
    )
    
    train_loader, train_scaler = get_dataloader(
        X_tr_task, Y_tr_task, 
        n_class=config.n_class, 
        scaler=train_scaler,
        batchsize=batch_size,  
        train=True
    )
    
    model.set_task(task_id)
    
    print(f"\n[Training Task {task_id+1}]")
    

    num_training_samples = len(X_tr_task)
    batches_per_epoch = int(np.ceil(num_training_samples / batch_size))
    n_iters = batches_per_epoch * n_epochs  # args.iters와 동일한 방식
    
    print(f"Training with {n_iters} iterations ({n_epochs} epochs equivalent)")
    print(f"Batches per epoch: {batches_per_epoch}")
    
    model.train()
    

    data_loader = iter(train_loader)
    iters_left = len(train_loader)
    

    pbar = tqdm(range(1, n_iters + 1), desc=f"Task {task_id+1}")
    
    for iteration in pbar:

        iters_left -= 1
        if iters_left == 0:
            data_loader = iter(train_loader)
            iters_left = len(train_loader)
        

        x, y = next(data_loader)
        x, y = x.to(device), y.to(device)
        

        loss, loss_consistency = model.observe(x, y, not_aug_inputs=x)
        
        if np.isnan(loss) or np.isnan(loss_consistency):
            print(f"[Warning] NaN loss at Task {task_id}, Iteration {iteration}")
            raise ValueError("NaN loss detected!")
        

        current_epoch = (iteration - 1) // batches_per_epoch + 1
        current_batch = (iteration - 1) % batches_per_epoch + 1
        pbar.set_postfix({
            'Epoch': f'{current_epoch}/{n_epochs}',
            'Batch': f'{current_batch}/{batches_per_epoch}',
            'Loss': f'{loss:.4f}',
            'Consistency': f'{loss_consistency:.4f}'
        })
        

        if iteration % batches_per_epoch == 0:
            print(f"\n==> Epoch [{current_epoch}/{n_epochs}] completed. "
                  f"Loss: {loss:.4f} | Consistency: {loss_consistency:.4f}")
    
    pbar.close()
    print(f"\n==> Task {task_id+1} training completed!")
    

    print(f"\n[Evaluation after Task {task_id+1}]")
    model.eval()
    
    with torch.no_grad():
        print(f"test_new, allowed_classes for ACC: {config.n_class}")
        

        accuracy = test_acc_tamil(config, model, acc_test_loaders[task_id], allowed_classes=config.n_class)
        ls_a.append(accuracy)
        
        accuracy_per_task = test_bwt_tamil(config, model, bwt_test_loaders[:task_id+1])
        accuracy_matrix.loc[
                            f"after task {task_id+1}",
                            accuracy_matrix.columns[:len(accuracy_per_task)]
                        ] = accuracy_per_task
    
    print(f"task {task_id} done")


def report_result(config, ls_a):
    print("\n" + "="*70)
    print("Final Results")
    print("="*70)
    print(f"\nAverage Accuracy: {np.mean(ls_a):.2f}%")
    print(f"The Accuracy for each tasks: {[f'{acc:.2f}' for acc in ls_a]}")

report_result(config, ls_a)

print("\nAccuracy matrix, based on all classes up to the task being evaluated")
print(accuracy_matrix)


def compute_fwt(R):
    N = R.shape[0]
    upper = np.triu(R, k=1)
    count = N * (N - 1) / 2
    fwt = upper.sum() / count
    return fwt

def compute_bwt(R):
    N = R.shape[0]
    total = 0.0
    count = 0
    for i in range(1, N):
        for j in range(i):
            total += (R[i, j] - R[j, j])
            count += 1
    bwt = total / count if count > 0 else 0.0
    return bwt

def compute_rem_and_bwt_plus(BWT):
    rem = 1 - abs(min(BWT, 0))
    bwt_plus = max(BWT, 0)
    return rem, bwt_plus

def compute_forgetting(R):
    N = R.shape[0]
    total = 0.0
    count = 0
    for j in range(N-1):
        max_acc = R[j, j]
        final_acc = R[N-1, j]
        total += (max_acc - final_acc)
        count += 1
    forgetting = total / count if count > 0 else 0.0
    return forgetting

R = accuracy_matrix.values
FWT = compute_fwt(R)
BWT = compute_bwt(R)
F = compute_forgetting(R)
REM, BWT_plus = compute_rem_and_bwt_plus(BWT)

print(f"\n=> FWT = {FWT}")
print(f"=> BWT = {BWT}")
print(f"=>  F = {F}")
print(f"=> REM = {REM}")
print(f"=> BWT+ = {BWT_plus}")


results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(results_dir, f'tamil_ember_{timestamp}.txt')

with open(results_file, "w") as f:
    f.write("Continual Learning Results (TAMIL-EMBER)\n")
    f.write("="*70 + "\n")
    f.write(f"Experiment time: {datetime.datetime.now()}\n")
    f.write(f"Seed: {seed}\n")
    f.write(f"Number of tasks: {n_tasks}\n")
    f.write(f"Initial classes: {initial_classes}\n")
    f.write(f"Increment classes: {increment_classes}\n")
    f.write(f"Buffer size: {args.buffer_size}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Training mode: Iteration-based (same as train_iter_cl)\n")
    f.write(f"Epochs equivalent: {n_epochs}\n")
    # 첫 task의 iteration 수 계산
    num_samples_task0 = len([y for y in Y_train if y < initial_classes])
    batches_task0 = int(np.ceil(num_samples_task0 / batch_size))
    n_iters_task0 = batches_task0 * n_epochs
    f.write(f"Iterations (task 0): {n_iters_task0} ({batches_task0} batches × {n_epochs} epochs)\n")
    f.write(f"Learning rate: {args.lr}\n")
    f.write(f"Reg weight: {args.reg_weight}\n")
    f.write(f"Beta: {args.beta}\n")
    f.write(f"EMA alpha: {args.ema_alpha}\n")
    f.write(f"EMA update freq: {args.ema_update_freq}\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Average Accuracy: {np.mean(ls_a):.2f}%\n")
    f.write(f"The Accuracy for each tasks: {ls_a}\n\n")
    
    f.write("Full Accuracy Matrix:\n")
    accuracy_matrix.to_csv(f, sep='\t', float_format='%.4f')
    
    f.write("\n\nContinual Learning Metrics:\n")
    f.write(f"FWT:  {FWT:.4f}\n")
    f.write(f"BWT:  {BWT:.4f}\n")
    f.write(f"F:    {F:.4f}\n")
    f.write(f"REM:  {REM:.4f}\n")
    f.write(f"BWT+: {BWT_plus:.4f}\n")

print(f"\n[Results saved to: {results_file}]")
print("\n" + "="*70)
print("Training completed successfully!")
print("="*70)