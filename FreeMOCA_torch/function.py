
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

    # print("class_pick_rand")
    print(class_arr)


    return Y_train, Y_test


def get_iter_train_dataset(x, y, n_class=None, n_inc=None, task=None):
   
   if task is not None:
    if task == 0:
       selected_indices = np.where(y < n_class)[0] 
    else:
       start = n_class - n_inc
       end = n_class
       selected_indices = np.where((y >= start) & (y < end))
    
    return x[selected_indices], y[selected_indices]

# The first 50,55,65, 70, .. are used
def get_iter_train_dataset_joint(x, y, n_class=None, n_inc=None, task=None):
    selected_indices = np.where(y < n_class)[0] 
    return x[selected_indices], y[selected_indices]

def get_iter_test_dataset(x, y, n_class):
    selected_indices = np.where(y < n_class)[0] 
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
    y_oh = oh(y_, num_classes=n_class)
    y_oh = torch.Tensor(y_oh)

    data_tensored = torch.utils.data.TensorDataset(x_, y_oh)
    if train: Loader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize, num_workers=1, sampler=sampler)
    else: Loader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize)
    
    return Loader, scaler

def get_iter_train_dataset_domain(x, y, task=None, config=None):
    # For domain-IL, return ALL classes for every task
    X = x.copy()
    Y = y.copy()

    # Apply DOMAIN SHIFT
    if config.domain_shift == "noise":
        noise_level = config.base_noise + task * config.noise_step
        X = X + np.random.normal(0, noise_level, X.shape)

    elif config.domain_shift == "scaling":
        scale = 1.0 + 0.1 * task
        X = X * scale

    elif config.domain_shift == "permutation":
        rng = np.random.RandomState(seed=task)
        perm = rng.permutation(X.shape[1])
        X = X[:, perm]

    return X, Y


#### New Metrics

# --- NEW: task-wise class range + task-wise test split ---
def get_task_class_range(init_classes, n_inc, task_id):
    """
    Task 0: classes [0, init_classes)
    Task t>0: classes [init_classes + (t-1)*n_inc, init_classes + t*n_inc)
    """
    if task_id == 0:
        return 0, init_classes
    start = init_classes + (task_id - 1) * n_inc
    end = init_classes + task_id * n_inc
    return start, end


def get_iter_test_dataset_task(x, y, init_classes, n_inc, task_id):
    """
    Return only the examples belonging to task_id (class-incremental split).
    """
    start, end = get_task_class_range(init_classes, n_inc, task_id)
    idx = np.where((y >= start) & (y < end))[0]
    return x[idx], y[idx]




def test(config, model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(config.device))
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            labels = labels.to(config.device)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    # print(f'Test Accuracy: {accuracy * 100:.2f}%')

    return accuracy*100




def measure_variance_collapse(prev_model, curr_model, interp_model, dataloader, device, layer_name='block2', alpha=0.5):
    """
    Diagnoses Variance Collapse by comparing the activation variance of the 
    interpolated model against the weighted average of the parent models.
    Uses the current task's dataloader as a probe (Memory-Free).
    """
    print(f"\n--- Diagnosing Variance Collapse on layer: {layer_name} ---")
    
    # 1. Setup Hooks to capture activations
    activations = {}
    
    def get_activation_hook(name):
        def hook(model, input, output):
            # output shape: [Batch, Channels, Length] or [Batch, Features]
            activations[name] = output.detach()
        return hook

    # Register hooks based on the layer name string
    # Assuming models have attributes like 'block1', 'block2', 'fc1' as seen in your models.py
    try:
        h1 = getattr(prev_model, layer_name).register_forward_hook(get_activation_hook('prev'))
        h2 = getattr(curr_model, layer_name).register_forward_hook(get_activation_hook('curr'))
        h3 = getattr(interp_model, layer_name).register_forward_hook(get_activation_hook('interp'))
    except AttributeError:
        print(f"Error: Layer '{layer_name}' not found in model.")
        return

    # 2. Set to Eval mode to disable Dropout (which adds random variance)
    prev_model.eval()
    curr_model.eval()
    interp_model.eval()

    # 3. Run a single batch (Probe)
    try:
        inputs, labels = next(iter(dataloader))
    except StopIteration:
        # Handle case if dataloader is empty
        print("Dataloader empty.")
        return

    inputs = inputs.float().to(device)
    
    # Forward pass (we only care about the hook capturing data)
    _ = prev_model(inputs)
    _ = curr_model(inputs)
    _ = interp_model(inputs)

    # 4. Calculate Variance
    # We calculate variance across the batch dimension (dim=0)
    # Then take the mean across all other dimensions (channels/features) to get a single scalar
    var_prev = torch.var(activations['prev'], dim=0).mean().item()
    var_curr = torch.var(activations['curr'], dim=0).mean().item()
    var_interp = torch.var(activations['interp'], dim=0).mean().item()

    # 5. Calculate Ratio
    # Expected variance if no collapse occurred (linear assumption)
    expected_var = (1 - alpha) * var_prev + alpha * var_curr
    
    # Avoid division by zero
    if expected_var < 1e-9:
        ratio = 1.0 
    else:
        ratio = var_interp / expected_var

    print(f"Variance Prev:   {var_prev:.4f}")
    print(f"Variance Curr:   {var_curr:.4f}")
    print(f"Variance Interp: {var_interp:.4f}")
    print(f"Expected Var:    {expected_var:.4f}")
    print(f"Collapse Ratio:  {ratio:.4f}")

    if ratio < 0.8:
        print(">> DIAGNOSIS: Significant Variance Collapse Detected!")
    elif ratio > 1.2:
        print(">> DIAGNOSIS: Variance Expansion (Rare)")
    else:
        print(">> DIAGNOSIS: No Significant Collapse (Warm-Start Successful)")

    # 6. Cleanup hooks
    h1.remove()
    h2.remove()
    h3.remove()
    
    return ratio    


