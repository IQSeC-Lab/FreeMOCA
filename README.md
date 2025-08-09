# FreeMOCA: Memory-Free Continual Learning for Malicious Code Analysis with Mode Connectivity and Interpolation

This repository contains code of the paper **FreeMOCA: Memory-Free Continual Learning for Malicious Code Analysis with Mode Connectivity and Interpolation**

FreeMOCA is a buffer-free continual learning framework that leverages mode connectivity for malicious code classification. After each task, FreeMOCA
computes a low-loss path that smoothly interpolates between the parameters of the current model and those of the previous task, preserving historical decision boundaries while adapting to new threats.
## Datasets

The dataset for the experiments can be downloaded from [here](https://drive.google.com/drive/folders/1YGmxcQGqu22ZQuZccpD81WUBKHh7c3Jq) and the [Zenodo Repository](https://zenodo.org/records/14537891).

### EMBER 2018 Dataset
We utilize the 2018 EMBER dataset, a benchmark known for its challenging malware classification tasks. Our study focuses on a subset of 337,035 malicious Windows PE files belonging to the top 100 malware families, each represented by more than 400 samples. The dataset includes features such as file size, PE and COFF header metadata, DLL characteristics, imported and exported functions, and various file properties (e.g., size and entropy). All features are processed using the feature hashing technique.

### AZ-Class Dataset
The AZ-Class dataset contains 285,582 samples from 100 Android malware families, with each family having at least 200 samples. We extract Drebin features (Arp et al., 2014) from the Android applications, covering eight categories including hardware access, permissions, API calls, and network addresses.
CUDA_VISIBLE_DEVICES=0 python none.py

## Environment

- PyTorch: 2.0.1  
- Conda: 4.7.12  
- Python: 3.8.13  
- GPU: NVIDIA RTX A6000  
- CUDA: 11.4  



## Experiments:
command lines are follows:

```bash
CUDA_VISIBLE_DEVICES=0 python none.py
CUDA_VISIBLE_DEVICES=0 python joint.py
CUDA_VISIBLE_DEVICES=0 python main.py
CUDA_VISIBLE_DEVICES=0 python none_KAN.py
CUDA_VISIBLE_DEVICES=0 python main_KAN.py
```




