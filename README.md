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

## Models

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

![Classifier Architecture](classifier_architecture.png)

---

### Convolutional Kolmogorov–Arnold Network (C-KAN)

- Uses KAN-based convolutions for enhanced representational power
- More sensitive to catastrophic forgetting than CNN
- FreeMOCA substantially mitigates forgetting in C-KAN models

---

## Experimental Setup

- **Learning scenario**: Class-Incremental Learning
- **Task sequence**:
  - Task 1: 50 classes
  - Tasks 2–11: +5 new classes per task
- **Optimizer**: SGD  
  - Learning rate: 1e-3  
  - Momentum: 0.9  
  - Weight decay: 1e-7
- **Batch size**: 256
- No replay, data augmentation, or rehearsal is used

---

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
