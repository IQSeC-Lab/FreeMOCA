# TAMIL-Ember/train.py

import torch
import sys
import numpy as np
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler

from models.tamil_ember import TAMIL_Ember
from models.buffer import Buffer
from models.tam import TAM
from models.tamil_ember import BackboneMLP
from data_ import get_ember_train_data, get_ember_test_data
from function import (Config, class_pick_rand, get_iter_train_dataset, 
                    get_iter_acc_test_dataset, get_iter_bwt_test_dataset, get_dataloader)

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

input_dim = 2381
hidden_dim = 512
feat_dim = 128
latent_dim = 64
code_dim = 64
n_tasks = 11
n_classes = 100

initial_classes = 50
increment_classes = 5
# buffer_size = 2000
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


data_dir = "./data"
torch.manual_seed(seed)
np.random.seed(seed)

print("\n[Data Loading]")
X_train, Y_train = get_ember_train_data(data_dir)
X_test, Y_test = get_ember_test_data(data_dir)

Y_train, Y_test, class_order = class_pick_rand(config, Y_train, Y_test)


# ae_type = 'relu', 'sigmoid', 'tanh'
print("\n[Model Initialization]")
backbone = BackboneMLP(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    feat_dim=feat_dim,
    n_tasks=n_tasks,
    code_dim=code_dim,
    n_classes=n_classes,
    ae_type='tanh'
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
transform = None

model = TAM(backbone, loss_fn, args, transform).to(device)
print(f"Model: TAM")
print(f"Device: {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# model = TAMIL_Ember(input_dim, hidden_dim, feat_dim, latent_dim, n_tasks, n_classes).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# buffer = Buffer(buffer_size=buffer_size, device=device, n_tasks=n_tasks)  # ring/reservoir 모드는 필요시 적용


task_accs = []
tasks = [f"task {i+1}" for i in range(n_tasks)]
rows = [f"after task {i+1}" for i in range(n_tasks)]
accuracy_matrix = pd.DataFrame(0.0, index=rows, columns=tasks)


task_sizes = []
for task_id in range(n_tasks):
    if task_id == 0:
        start = 0
        end = initial_classes
    else:
        start = initial_classes + (task_id - 1) * increment_classes
        end = initial_classes + task_id * increment_classes
    
    task_indices = np.where((Y_train >= start) & (Y_train < end))[0]
    task_sizes.append(len(task_indices))

task_sizes = np.array(task_sizes)
print(f"task_sizes: {task_sizes}")


scaler = StandardScaler()  


for task_id in range(n_tasks):
    print(f"\n==== Task {task_id} ====")

    n_classes_so_far = initial_classes + task_id * increment_classes

    X_tr_task, Y_tr_task = get_iter_train_dataset(
        X_train, Y_train, 
        n_class=n_classes_so_far, 
        n_inc=increment_classes, 
        task=task_id
    )
    

    train_loader, scaler = get_dataloader(
                    X_tr_task, Y_tr_task, 
                    n_class=n_classes_so_far, scaler=scaler,
                    batchsize=batch_size,  train=True)


    model.set_task(task_id)


    print(f"\n[Training Task {task_id+1}]")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_consistency_loss = 0.0
        n_batches = 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            loss, loss_consistency = model.observe(x, y, not_aug_inputs=x)


            if np.isnan(loss) or np.isnan(loss_consistency):
                print(f"[Warning] NaN loss encountered at Task {task_id}, Epoch {epoch+1}, Step {step+1}")
                print(f"Loss: {loss}, Consistency: {loss_consistency}")
                raise ValueError("NaN loss detected!")
            
            epoch_loss += loss
            epoch_consistency_loss += loss_consistency
            n_batches += 1

            # if (step+1)%50 == 0:
            #     avg_loss = epoch_loss / n_batches
            #     avg_consistency_loss = epoch_consistency_loss / n_batches
            #     print(f"  Epoch [{epoch+1}/{n_epochs}] Step [{step+1}/{len(train_loader)}] "
            #           f"Loss: {avg_loss:.4f} | Consistency: {avg_consistency_loss:.4f}")

        # epoch summary        
        avg_epoch_loss = epoch_loss / n_batches
        avg_epoch_consistency_loss = epoch_consistency_loss / n_batches
        print(f"==> Epoch [{epoch+1}/{n_epochs}] completed. "
              f"Avg Loss: {avg_epoch_loss:.4f} | Avg Consistency: {avg_epoch_consistency_loss:.4f}")        


    print(f"\n[Evaluation after Task {task_id}]")
    model.eval()
    
    with torch.no_grad():

        print(f" Evaluating on Task (Classes 0 to {n_classes_so_far-1})")

        X_te_cumulative, Y_te_cumulative = get_iter_acc_test_dataset(
            X_test, Y_test, 
            n_class=n_classes_so_far
        )

        test_loader_cumulative, _ = get_dataloader(
            X_te_cumulative, Y_te_cumulative, 
            n_class=n_classes_so_far, 
            scaler=scaler,
            batchsize=batch_size, 
            train=False
        )  

        allowed_classes_cum = list(range(n_classes_so_far))
        allowed_classes_cum = torch.tensor(allowed_classes_cum, device=device, dtype=torch.long)

        correct_cumulative = 0
        total_cumulative = 0

        for x, y in test_loader_cumulative:
            x, y = x.to(device), y.to(device)
                
            logits = model.forward(x)


            mask = torch.full_like(logits, float('-inf'))
            mask[:, allowed_classes_cum] = 0.0
            masked_logits = logits + mask

            pred = masked_logits.argmax(1)
            correct_cumulative += (pred == y).sum().item()
            total_cumulative += y.size(0)
            
        acc_cumulative = correct_cumulative / total_cumulative if total_cumulative > 0 else 0.0
        task_accs.append(acc_cumulative) 
        # accuracy_matrix.loc[f"after task {task_id+1}", f"task {eval_task_id+1}"] = acc_cumulative
        print(f"Using Cumulative range -> Accuracy: {acc_cumulative*100:.2f}% ({correct_cumulative}/{total_cumulative})")

        print(f"Per-task evaluation: ")
        
        for eval_task_id in range(task_id + 1):           
            n_classes_eval = initial_classes + eval_task_id * increment_classes
            
            X_te, Y_te = get_iter_bwt_test_dataset(
                X_test, Y_test, 
                n_class=n_classes_eval,
                n_inc=increment_classes, 
                task=eval_task_id
            )

            test_loader, _ = get_dataloader(
                X_te, Y_te, 
                n_class=n_classes, scaler=scaler,
                batchsize=batch_size, train=False)
            
            # allowed classes for per-task evaluation
            allowed_classes = list(range(n_classes_eval))
            allowed_classes = torch.tensor(allowed_classes, device=device, dtype=torch.long)

            correct = 0
            total = 0

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                logits = model.forward(x)

                mask = torch.full_like(logits, float('-inf'))
                mask[:, allowed_classes] = 0.0
                masked_logits = logits + mask
                
                pred = masked_logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            
            acc = correct / total if total > 0 else 0.0
            accuracy_matrix.loc[f"after task {task_id+1}", f"task {eval_task_id+1}"] = acc

        print(f"Task {task_id} done\n")


print("\n" + "="*70)
print("Final Results")
print("="*70)

print("\n[Task-wise Accuracy]")
for t, acc in enumerate(task_accs):
    weight = task_sizes[t] / np.sum(task_sizes)
    print(f"Task {t+1}: {acc*100:.2f}% (samples: {task_sizes[t]:,}, weight: {weight*100:.2f}%)")

simple_avg = np.mean(task_accs)
weighted_avg = np.sum(np.array(task_accs) * task_sizes) / np.sum(task_sizes)
print(f"\nSimple Average:   {simple_avg*100:.2f}%")
print(f"Weighted Average: {weighted_avg*100:.2f}%")


print("\n[Full Accuracy Matrix]")
print(accuracy_matrix)


def compute_fwt(R):
    """
    Forward Transfer (FWT)
    R[i, j] is test accuracy on task j after training task i.
    FWT averages all entries ABOVE the diagonal.
    """
    N = R.shape[0]
    # Upper triangular without diagonal
    upper = np.triu(R, k=1)
    # Count non-zero entries (valid FWT spots)
    count = N * (N - 1) / 2
    fwt = upper.sum() / count
    return fwt


def compute_bwt(R):
    """
    Backward Transfer (BWT)
    Measures how learning new tasks influences past tasks.
    Formula: avg over i>j of (R[i, j] - R[j, j])
    """
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
    """
    Splits BWT into:
    - REM = Remembering = 1 − |min(BWT, 0)|
    - BWT+ = positive backward transfer = max(BWT, 0)
    These follow the paper's mapping to [0,1].
    """
    rem = 1 - abs(min(BWT, 0))   # forgetting (negative BWT) mapped to [0,1]
    bwt_plus = max(BWT, 0)       # positive improvements
    return rem, bwt_plus

R = accuracy_matrix.values

fwt = compute_fwt(R)
bwt = compute_bwt(R)
rem, bwt_plus = compute_rem_and_bwt_plus(bwt)

print("\n[FWT & BWT Metrics]")
print(f"Forward Transfer (FWT):  {fwt:.4f}")
print(f"Backward Transfer (BWT): {bwt:.4f}")
print(f"Remembering (REM):       {rem:.4f}")
print(f"BWT+:                    {bwt_plus:.4f}")


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
    f.write(f"Epochs per task: {n_epochs}\n")
    f.write(f"Learning rate: {args.lr}\n")
    f.write(f"Reg weight: {args.reg_weight}\n")
    f.write(f"Beta: {args.beta}\n")
    f.write(f"EMA alpha: {args.ema_alpha}\n")
    f.write(f"EMA update freq: {args.ema_update_freq}\n")
    f.write("="*70 + "\n\n")
    
    f.write("Task-wise Accuracy (Cumulative Evaluation):\n")
    for t, acc in enumerate(task_accs):
        weight = task_sizes[t] / np.sum(task_sizes)
        f.write(f"Task {t+1}: {acc*100:.2f}% (samples: {task_sizes[t]:,}, weight: {weight*100:.2f}%)\n")
    
    f.write(f"\nSimple Average:   {simple_avg*100:.2f}%\n")
    f.write(f"Weighted Average: {weighted_avg*100:.2f}%\n\n")
    
    f.write("Full Accuracy Matrix:\n")
    # pandas DataFrame으로 저장 (row/column 이름 포함)
    accuracy_matrix.to_csv(f, sep='\t', float_format='%.4f')
    
    f.write("\n\nContinual Learning Metrics:\n")
    f.write(f"Forward Transfer (FWT):  {fwt:.4f}\n")
    f.write(f"Backward Transfer (BWT): {bwt:.4f}\n")
    f.write(f"Remembering (REM):       {rem:.4f}\n")
    f.write(f"BWT+:                    {bwt_plus:.4f}\n")

print(f"\n[Results saved to: {results_file}]")
print("\n" + "="*70)
print("Training completed successfully!")
print("="*70)
