# FreeMOCA
## Memory-Free Continual Learning for Malicious Code Analysis with Mode Connectivity and Interpolation

Official implementation of the paper  
**FreeMOCA: Memory-Free Continual Learning for Malicious Code Analysis with Mode Connectivity and Interpolation**

---

## Overview

FreeMOCA is a **buffer-free continual learning framework** for malware classification that mitigates **catastrophic forgetting** without storing or replaying past data. Instead of using memory buffers or generative replay, FreeMOCA exploits **mode connectivity** in parameter space to preserve previously learned knowledge while adapting to new malware families.

After each task, FreeMOCA computes a **low-loss interpolation path** between the parameters of the current task model and the previous task model. This interpolation preserves historical decision boundaries while enabling efficient adaptation to evolving malware distributions.

FreeMOCA is designed for **real-world malware detection systems**, where memory, privacy, and computational constraints make replay-based continual learning impractical.

---

## Key Features

- **Memory-free continual learning** (no replay buffers or synthetic data)
- **Mode connectivity–based interpolation** between task-specific optima
- **Warm-start training** to align task minima in parameter space
- **Adaptive layer-wise interpolation** based on parameter update magnitude
- Supports **CNN** and **Convolutional Kolmogorov–Arnold Networks (C-KAN)**
- Evaluated on **large-scale Windows and Android malware datasets**

---

## Method Summary

FreeMOCA operates in a **class-incremental learning (Class-IL)** setting:

1. Tasks arrive sequentially, each introducing new malware families.
2. The model for each task is initialized from the previous task (warm start).
3. After training, a **low-loss interpolation** is computed between the current and previous task parameters.
4. The interpolated model replaces the current model and is used for the next task.
5. No past data, features, or generated samples are stored.

This process constructs a chain of connected solutions that lie on a low-loss manifold, significantly reducing catastrophic forgetting.

---

## Datasets

### EMBER-Class (Windows Malware)

- Derived from the **EMBER 2018** dataset
- **337,035** malicious PE files
- **100 malware families**, each with more than 400 samples
- Feature categories:
  - PE and COFF headers
  - Imported and exported functions
  - DLL characteristics
  - File properties (size, entropy, metadata)
- Features are encoded using **feature hashing**

Dataset sources:
- EMBER: https://github.com/elastic/ember  
- Androzoo https://zenodo.org/records/14537891

---

### AZ-Class (Android Malware)

- Collected from the **AndroZoo** repository
- **285,582** Android malware samples
- **100 malware families**, each with at least 200 samples
- Features extracted using **DREBIN**, including:
  - Permissions
  - API calls
  - Hardware components
  - Network addresses

---

## Model

We evaluate FreeMOCA using two classifiers.

### Convolutional Neural Network (CNN)

- Two convolutional blocks:
  - **Block 1**:
    - Two Conv1D layers (kernel size = 3, stride = 3)
    - Batch normalization, ReLU, dropout (0.5), max pooling
  - **Block 2**:
    - One Conv1D layer (kernel size = 3, stride = 2)
    - Batch normalization, ReLU, dropout
- Fully connected classification head with softmax output

---



## Running FreeMOCA (Class-IL & Domain-IL)

This repository supports two continual learning scenarios:

- **Class-Incremental Learning (Class-IL)** → new classes are introduced over tasks.
- **Domain-Incremental Learning (Domain-IL)** → label space is fixed, but data distribution shifts across domains (e.g., months/years).

FreeMOCA interpolation is applied after each task using adaptive layer-wise interpolation.

---

###  Common Arguments

| Argument | Description |
|-----------|-------------|
| `--scenario {class,domain}` | Continual learning scenario (**required**) |
| `--dataset {EMBER,AZ}` | Dataset name (**required**) |
| `--nb_task` | Number of sequential tasks (**required**) |
| `--init_classes` | Number of classes at task 0 (**required**) |
| `--epochs` | Epochs per task |
| `--batchsize` | Batch size |
| `--lr` | Learning rate |
| `--momentum` | SGD momentum |
| `--weight_decay` | Weight decay |
| `--lambda_min` | Minimum interpolation weight |
| `--lambda_max` | Maximum interpolation weight |

---

### 1. Class-Incremental Learning (Class-IL)

In Class-IL:
- The classifier head expands at each task.
- Classes are shuffled once at the beginning.
- Continual learning metrics (ACC, BWT, FWT) are computed.


#### EMBER

TRAIN_DATA/
├── XY_train.npz
└── XY_test.npz


### AZ

TRAIN_DATA/
├── AZ_Class_Train.npz
└── AZ_Class_Test.npz
---

---

## ⚙️ Required Class-IL Arguments

- `--n_inc` (must be > 0)
- `--final_classes`
- `--train_data`
- `--test_data`

---



```bash
python main.py \
  --scenario class \
  --dataset EMBER \
  --nb_task 11 \
  --init_classes 50 \
  --n_inc 5 \
  --final_classes 100 \
  --train_data /path/to/EMBER_Class \
  --test_data /path/to/EMBER_Class \
  --epochs 50 \
  --batchsize 256 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight_decay 1e-6 \
  --lambda_min 0.3 \
  --lambda_max 0.7


python main.py \
  --scenario class \
  --dataset AZ \
  --nb_task 11 \
  --init_classes 50 \
  --n_inc 5 \
  --final_classes 100 \
  --train_data /path/to/AZ_Class \
  --test_data /path/to/AZ_Class \
  --epochs 50 \
  --batchsize 256 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight_decay 1e-6 \
  --lambda_min 0.3 \
  --lambda_max 0.7



## Baselines

- **None**: Train only on the current task (lower bound)
- **Joint**: Train on all data seen so far (upper bound, impractical)

### Compared Continual Learning Methods

- iCaRL
- Experience Replay (ER)
- Generative Replay (GR)
- BI-R
- TAMiL
- MaLCL
- CLEWI
- PEARL

FreeMOCA outperforms these methods **without storing any past data**.

---

## Results Highlights

- **EMBER-Class**:
  - ~10% absolute improvement over strongest baselines
  - Up to **46% reduction in catastrophic forgetting**
- **AZ-Class**:
  - ~5% improvement over state-of-the-art methods
- **C-KAN**:
  - Accuracy improves from ~13–18% to over **55–64%**

---

## Installation

```bash
conda create -n freemoca python=3.9
conda activate freemoca
pip install -r requirements.txt
